"""
A physics check for a piece the moment it is released, run in a disposable scene that
holds only the shape-sorting board and that one piece -- no robot -- so what
:mod:`segmind` detects there says what the release actually did, without ever touching
the belief the caller keeps.

:mod:`~experiments.tracy_experiments.pickup.pickup_demo_simulated` (and, later, the
real-robot pickup demo) leave a released piece exactly where the plan let go of it: the
belief has no physics, so nothing tells it whether the piece fell through its hole, came
to rest on the board's lid, or was disturbed on the way. :class:`ReleaseCheck` answers
that by dropping a piece of the same kind, at the exact pose the belief has it at, into
its own throwaway :class:`~experiments.tracy_experiments.montessori.world.
TracyMontessoriWorld` -- built without ever mounting a robot -- and reading back what
:mod:`segmind` detected while it settled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List, Tuple

from experiments.montessori.event_monitoring import (
    MontessoriEventMonitor,
    build_shape_monitor_in_scene,
)
from experiments.montessori.pieces import KnownPiece, KnownPieceSet, SMALLER_PIECES
from experiments.montessori.scenarios import (
    SETTLING_LIMIT,
    SETTLING_WINDOW,
    SIMULATION_STEP_SIZE,
    SortingScene,
    STILLNESS,
)
from experiments.montessori.semantics import (
    MontessoriShape,
    MontessoriShapeCategory,
    ShapeSortingBoard,
)
from experiments.tracy_experiments.equipment import (
    parse_tracy,
    tracy_table_mount_position,
)
from experiments.tracy_experiments.montessori.scene_builder import (
    TRACY_MOUNT_X,
    TRACY_MOUNT_Y,
)
from experiments.tracy_experiments.montessori.world import TracyMontessoriWorld
from experiments.tracy_experiments.real_time_simulation import RealTimeSimulation
from krrood.exceptions import DataclassException
from segmind.datastructures.events import ContainmentEvent, DetectionEvent
from segmind.detectors.atomic_event_detectors_nodes import MotionDetector
from semantic_digital_twin.adapters.multi_sim import MujocoSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

FLOOR_SCALE = Scale(2.0, 2.0, 0.05)
"""
Size of the solid slab put under the board, wide enough to reach past a loose piece
released anywhere on the table.

:class:`~experiments.tracy_experiments.montessori.world.TracyMontessoriWorld`'s own
``"floor"`` body is a visual reference only, real Tracy's own (excluded) table being what
normally stands under it physically; without that table a piece that misses the board
altogether, or clears a hole it fits through, would otherwise fall forever.
"""

FLOOR_CLEARANCE_ABOVE_TABLE_TOP = 0.01
"""
How far above the table height the check floor's own top surface sits, in metres.

A hole's landing region runs from the table height up to the board's own top surface, so
a floor placed exactly at the table height leaves a resting piece's underside right at
that region's own lower boundary -- and the brief bounce a real drop settles out of can
dip it fractionally past that boundary, under :attr:`~segmind.detectors.
spatial_relation_detector_nodes.BaseContainmentDetector.containment_threshold`, before
it settles back. Raising the floor this far gives the settle room to stay inside the
landing region throughout, confirmed against every piece in the set; it is small enough
that a piece still has to have gone through the hole to reach it.
"""

MOTION_DETECTOR_CATCH_UP_TICKS = MotionDetector.window_size
"""
How many more settling windows :meth:`ReleaseCheck._settle` keeps ticking the monitor
for once it finds the piece still.

:class:`~segmind.detectors.atomic_event_detectors_nodes.MotionDetector` (the base of
:class:`~segmind.detectors.atomic_event_detectors_nodes.StopTranslationDetector`, which
:class:`~segmind.datastructures.events.PlacingEvent` is built on) only concludes a piece
has stopped once its own sliding window of this many ticks is entirely past ones where
the piece was moving; tied to that class's own default rather than restated, since it is
that window this is catching up to.
"""

# %% the outcome


@dataclass(frozen=True)
class ReleaseOutcome:
    """
    What a disposable physics settle of one released piece showed.
    """

    events: List[DetectionEvent]
    """
    Every event :mod:`segmind` detected while the piece settled.
    """

    settled_pose: Pose
    """
    Where the piece came to rest, in the check world's own root frame -- numerically the
    same frame the belief's piece poses are expressed in, since the check world is built
    the same way, without ever mounting a robot.
    """

    fell_through: bool
    """
    Whether the piece came to rest inside its hole's landing region, read from a
    :class:`~segmind.datastructures.events.ContainmentEvent` :attr:`events` carries for
    it.
    """


@dataclass
class NoKnownPieceOfCategoryError(DataclassException):
    """
    Raised when :attr:`ReleaseCheck.piece_set` has no piece of the category being
    checked.
    """

    category: MontessoriShapeCategory
    """
    The category no piece in the set answers to.
    """

    def error_message(self) -> str:
        return "No piece of category %s in the piece set." % self.category.value

    def suggest_correction(self) -> str:
        return (
            "Check the piece against a ReleaseCheck built with the same piece set the "
            "scene was sorted from."
        )


# %% the check


@dataclass
class ReleaseCheck:
    """
    Drops a released piece into a throwaway copy of the shape-sorting scene -- the board
    and that one piece, no robot -- and reports what :mod:`segmind` saw.

    The scene is rebuilt fresh for every :meth:`simulate` call rather than reused, so
    one release's settle can never see an earlier one's disturbance; the piece being
    checked is the only physics run for now.
    """

    piece_set: KnownPieceSet = field(default_factory=lambda: SMALLER_PIECES)
    """
    Where the checked piece's own geometry (mesh, height) is looked up by shape
    category; the same set the scene being checked was sorted from.
    """

    settling_window: float = SETTLING_WINDOW
    """
    Simulated seconds advanced between two stillness checks.
    """

    stillness: float = STILLNESS
    """
    How far the piece may travel over one :attr:`settling_window` and still count as
    having come to rest.
    """

    settling_limit: float = SETTLING_LIMIT
    """
    Simulated seconds the piece is given to come to rest before the settle gives up and
    reports wherever it has gotten to.
    """

    step_size: float = SIMULATION_STEP_SIZE
    """
    The physics time step, in seconds.
    """

    def simulate(
        self, piece: MontessoriShape, board: ShapeSortingBoard
    ) -> ReleaseOutcome:
        """
        Drop a piece of the same kind as ``piece``, at ``piece``'s own current pose,
        into a fresh scene holding only ``board``'s own hole layout -- stood at
        ``board``'s own measured pose -- and that piece, and settle it under gravity
        while :mod:`segmind` watches.

        :param piece: The belief's own piece, read for its shape category and its
            current pose; never modified.
        :param board: The belief's own board, read for its hole layout and its current
            pose; never modified. A perceived board's own body has no open-hole
            collision or landing regions to check against, so only its layout and pose
            are used -- the check drops into :class:`~experiments.tracy_experiments.
            montessori.world.TracyMontessoriWorld`'s own modelled board, stood where
            ``board`` is.
        :raises NoKnownPieceOfCategoryError: If :attr:`piece_set` has no piece of
            ``piece``'s category.
        """
        world, checked_piece = self._build_check_world(piece, board)
        monitor = build_shape_monitor_in_scene(world, checked_piece)
        connection = checked_piece.root.parent_connection
        with RealTimeSimulation(
            world=world,
            headless=True,
            paced_to_the_wall_clock=False,
            step_size=self.step_size,
            physically_simulated_dofs=set(connection.passive_dofs),
            # An unpaced advance can run many simulated seconds in a fraction of a
            # real one, too little wall-clock time for the default throttled sync to
            # ever read the pose back -- see RealTimeSimulation.sync_rate_hz.
            sync_rate_hz=MujocoSynchronizer.UNTHROTTLED_SYNC_RATE_HZ,
        ) as simulation:
            monitor.start()
            self._settle(simulation, world, checked_piece, monitor)
            monitor.stop()

        world.update_forward_kinematics()
        settled_pose = checked_piece.root.global_transform.to_pose()
        fell_through = any(
            isinstance(event, ContainmentEvent)
            and event.tracked_object is checked_piece.root
            for event in monitor.events
        )
        return ReleaseOutcome(
            events=monitor.events, settled_pose=settled_pose, fell_through=fell_through
        )

    def _settle(
        self,
        simulation: RealTimeSimulation,
        world: World,
        checked_piece: MontessoriShape,
        monitor: MontessoriEventMonitor,
    ) -> None:
        """
        Advance the simulation until :attr:`checked_piece` stops moving, or
        :attr:`settling_limit` runs out, ticking ``monitor`` after every advance.

        Keeps ticking :data:`MOTION_DETECTOR_CATCH_UP_TICKS` windows past the moment
        :attr:`checked_piece` is found still: :class:`~segmind.detectors.
        atomic_event_detectors_nodes.StopTranslationDetector` only concludes a piece has
        stopped once its own, coarser sliding window is *entirely* past ones where the
        piece was still moving, which stillness found here alone does not guarantee --
        without this, ``monitor`` can stop being ticked before that window has caught up,
        and neither :class:`~segmind.datastructures.events.StopTranslationEvent` nor the
        :class:`~segmind.datastructures.events.PlacingEvent` built on it ever fires.
        """
        elapsed = 0.0
        previous_position = self._position_of(world, checked_piece)
        while elapsed < self.settling_limit:
            simulation.advance(self.settling_window)
            monitor.tick()
            elapsed += self.settling_window
            position = self._position_of(world, checked_piece)
            if np.linalg.norm(position - previous_position) < self.stillness:
                self._catch_up_the_motion_detectors(simulation, monitor)
                return
            previous_position = position

    def _catch_up_the_motion_detectors(
        self, simulation: RealTimeSimulation, monitor: MontessoriEventMonitor
    ) -> None:
        """
        Keep advancing and ticking ``monitor`` for
        :data:`MOTION_DETECTOR_CATCH_UP_TICKS` more windows, so every tick
        :class:`~segmind.detectors.atomic_event_detectors_nodes.MotionDetector`'s own
        sliding window holds was taken after the piece was already found still.
        """
        for _ in range(MOTION_DETECTOR_CATCH_UP_TICKS):
            simulation.advance(self.settling_window)
            monitor.tick()

    @staticmethod
    def _position_of(world: World, checked_piece: MontessoriShape) -> np.ndarray:
        """
        :return: Where ``checked_piece`` currently stands, in the world root frame.
        """
        return (
            SortingScene(world)
            .position_of(checked_piece.shape_category)
            .to_np()
            .flatten()[:3]
        )

    def _build_check_world(
        self, piece: MontessoriShape, board: ShapeSortingBoard
    ) -> Tuple[World, MontessoriShape]:
        """
        Build the throwaway scene: a solid floor at Tracy's own table height, the
        modelled board standing at ``board``'s own measured pose, and one movable piece
        of ``piece``'s own kind standing at ``piece``'s own current pose.

        No robot is mounted -- the table height alone is read off a parsed-but-unmounted
        Tracy, exactly as :class:`~experiments.tracy_experiments.montessori.
        scene_builder.TracyOnItsOwnTable` reads it before mounting one.
        """
        tracy = parse_tracy()
        _, table_top_z = tracy_table_mount_position(
            tracy, x=TRACY_MOUNT_X, y=TRACY_MOUNT_Y
        )
        known_piece = self._known_piece_of(piece.shape_category)
        montessori = TracyMontessoriWorld(
            shapes_are_movable=True,
            table_top_z=table_top_z,
            pieces=KnownPieceSet(pieces=(known_piece,)),
        )
        world = montessori.world
        self._add_floor(world, table_top_z)
        self._relocate_board(world, montessori.board, board)
        [checked_piece] = world.get_semantic_annotations_by_type(MontessoriShape)
        world.move_branch_to(
            checked_piece.root,
            HomogeneousTransformationMatrix(
                piece.root.global_transform.to_np(), reference_frame=world.root
            ),
        )
        return world, checked_piece

    @staticmethod
    def _relocate_board(
        world: World,
        modelled_board: ShapeSortingBoard,
        measured_board: ShapeSortingBoard,
    ) -> None:
        """
        Move ``modelled_board`` -- and every one of its holes' landing regions -- from
        where it was built to ``measured_board``'s own current pose.

        A hole itself hangs off the board's own body, so moving the board alone carries
        it along; a landing region hangs off ``world``'s root independently of both and
        needs the same rigid transform applied to it by hand, or it would be left
        behind.

        :param world: The world ``modelled_board`` stands in, modified in place.
        :param modelled_board: The check world's own, properly-holed board, built at a
            default pose.
        :param measured_board: The board whose pose ``modelled_board`` is moved to; may
            belong to a different world (only its pose is read).
        """
        default_pose = HomogeneousTransformationMatrix(
            modelled_board.root.global_transform.to_np(), reference_frame=world.root
        )
        measured_pose = HomogeneousTransformationMatrix(
            measured_board.root.global_transform.to_np(), reference_frame=world.root
        )
        transform = measured_pose @ default_pose.inverse()

        # Read every landing region's own default pose before moving anything: unlike a
        # hole, it is not a descendant of the board, so it would not otherwise be
        # affected by moving the board first.
        landing_regions_at_their_default_pose = [
            (
                hole.landing_region,
                HomogeneousTransformationMatrix(
                    hole.landing_region.global_transform.to_np(),
                    reference_frame=world.root,
                ),
            )
            for hole in modelled_board.apertures
            if hole.landing_region is not None
        ]

        world.move_branch_to(modelled_board.root, measured_pose)
        for region, default_region_pose in landing_regions_at_their_default_pose:
            world.move_branch_to(region, transform @ default_region_pose)

    @staticmethod
    def _add_floor(world: World, table_top_z: float) -> None:
        """
        Add a solid :data:`FLOOR_SCALE` slab under the board, its top face
        :data:`FLOOR_CLEARANCE_ABOVE_TABLE_TOP` above ``table_top_z``.

        :param world: The world to add the floor to, modified in place.
        :param table_top_z: Height of the table surface the board stands on.
        """
        floor_top_z = table_top_z + FLOOR_CLEARANCE_ABOVE_TABLE_TOP
        floor = Body(
            name=PrefixedName(name="release_check_floor", prefix="montessori"),
            collision=ShapeCollection([Box(scale=FLOOR_SCALE)]),
        )
        with world.modify_world():
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=floor,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=floor_top_z - FLOOR_SCALE.z / 2, reference_frame=world.root
                    ),
                )
            )

    def _known_piece_of(self, category: MontessoriShapeCategory) -> KnownPiece:
        """
        :return: :attr:`piece_set`'s own piece of ``category``.
        :raises NoKnownPieceOfCategoryError: If it has none.
        """
        for known_piece in self.piece_set.pieces:
            if known_piece.category is category:
                return known_piece
        raise NoKnownPieceOfCategoryError(category=category)
