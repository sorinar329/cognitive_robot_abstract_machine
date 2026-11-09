#!/usr/bin/env python
"""The IJCAI expo demo: the Franka Emika Panda stacks four cubes into a tower
in MuJoCo, diagnoses every failed attempt against the trees fitted on
successful attempts, and recovers with the perfect parameters.

Each cube's first attempt samples its pick-up and place parameters from a range wider
than the successful attempts used, with a friction too low for the fingers to hold the
cube, so the first attempt fails. After the failure both actions are
diagnosed (:mod:`causal_diagnosis`), the anomalous parameters are reported, and every
later attempt uses the perfect value of every parameter. The run is shown live in the
cramera viewer.

Run it from this folder::

    python inference3b.py --iterations 3

and open the viewer with ``cramera`` (from ``~/cramera``), which follows the live run.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree

import mujoco
import numpy as np

from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor, PlanExecutor
from coraplex.plans.failures import PlanFailure
from coraplex.exceptions import MotionDidNotFinish
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramera_bridge.coraplex_backend import CrameraVisualization
from cramera_bridge.live.world_state import MeasuredJointPositions
from cramera_bridge.visualization import LiveVisualization
from cramera_protocol.diagnoses import (
    DiagnosedStep,
    DiagnosisStatus,
    ParameterCorrection,
    ParameterVerdict,
)
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from physics_simulators.base_simulator import SimulatorCallbackResult
from segmind.event_segmentation import Segmind
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.contact import ContactParameters
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.world_entity import Body

from causal_diagnosis import ActionCausalDiagnoser, ActionDiagnosis, DiagnosedAction

# %% the scene


class CubeName(StrEnum):
    """The cubes, in the order they are stacked: each onto the one before
    it."""

    CUBE_0 = "cube0"
    CUBE_1 = "cube1"
    CUBE_2 = "cube2"
    CUBE_3 = "cube3"


@dataclass
class CubeSpawn:
    """Where a cube stands on the floor before it is stacked, and its
    colour."""

    name: CubeName
    """The cube."""

    x: float
    """How far in front of the robot it stands."""

    y: float
    """Where it stands sideways."""

    color: Color
    """The colour it is drawn in."""


class UrdfElement(StrEnum):
    """The elements of a URDF description that tell what a link is drawn and
    collides with."""

    LINK = "link"
    VISUAL = "visual"
    COLLISION = "collision"


@dataclass
class VisualDescription:
    """A URDF description reduced to what is drawn of it: nothing of it
    collides, and a link that only collides is drawn as it collides."""

    path: Path
    """The URDF file."""

    def text(self) -> str:
        """:return: The description without collision elements."""
        root = ElementTree.parse(self.path).getroot()
        for link in root.iter(UrdfElement.LINK):
            collisions = link.findall(UrdfElement.COLLISION)
            if not link.findall(UrdfElement.VISUAL):
                for collision in collisions:
                    collision.tag = UrdfElement.VISUAL
                continue
            for collision in collisions:
                link.remove(collision)
        return ElementTree.tostring(root, encoding="unicode")


# %% the parameters


@dataclass
class ParameterPrior:
    """A normal distribution of one parameter, truncated to the range it is
    sampled in."""

    mean: float
    """The mean of the distribution."""

    standard_deviation: float
    """The spread of the distribution."""

    lowest: float
    """The smallest value sampled."""

    highest: float
    """The largest value sampled."""

    def sample(self, generator: np.random.Generator) -> float:
        """
        :return: A value of the truncated distribution.
        """
        while True:
            value = generator.normal(self.mean, self.standard_deviation)
            if self.lowest <= value <= self.highest:
                return float(value)


class ActionField(Enum):
    """The trees' names of the parameters whose action field is named
    differently."""

    FINAL_APPROACH = ("grasp_linear_velocity", "final_approach_linear_velocity")
    STALL_TIME = ("grasp_stall_min_time", "grasp_stall_minimum_time")

    @classmethod
    def of(cls, parameter_name: str) -> str:
        """
        :param parameter_name: The name of a parameter among the trees' variables.
        :return: The name of the action field it sets.
        """
        for member in cls:
            if member.value[0] == parameter_name:
                return member.value[1]
        return parameter_name


@dataclass
class StackingParameters:
    """The parameters of one attempt's pick-up and place, by their names among
    the trees' variables."""

    pick_up: dict[str, float]
    """The pick-up's parameters, the cube's friction among them."""

    place: dict[str, float]
    """The place's parameters."""

    @property
    def friction(self) -> float:
        """The sliding friction the picked cube is given."""
        return self.pick_up["object_friction"]

    def pick_up_fields(self) -> dict[str, float]:
        """
        :return: The pick-up's parameters by the name of the action field they set.
        """
        return {
            ActionField.of(name): value
            for name, value in self.pick_up.items()
            if name != "object_friction"
        }

    def place_fields(self) -> dict[str, float]:
        """
        :return: The place's parameters by the name of the action field they set.
        """
        return {ActionField.of(name): value for name, value in self.place.items()}

    def described(self) -> str:
        """
        :return: One line listing every parameter.
        """
        return ", ".join(
            f"{name}={value:.4f}"
            for name, value in {**self.pick_up, **self.place}.items()
        )


def wide_pick_up_priors() -> dict[str, ParameterPrior]:
    """
    :return: Priors of a first attempt, reaching past the range the successful
        attempts used. The friction is always too low for the fingers to lift the
        cube, which slides out of them where it stands: at 0.08 they already carry it
        part of the way and drop it anywhere, and faster or slower motions alone do not
        make a pick-up fail.
    """
    return {
        "pre_approach_linear_velocity": ParameterPrior(0.2, 0.08, 0.04, 0.35),
        "grasp_linear_velocity": ParameterPrior(0.16, 0.09, 0.01, 0.35),
        "grasp_closing_velocity": ParameterPrior(0.2, 0.1, 0.02, 0.4),
        "grasp_stall_min_time": ParameterPrior(0.4, 0.15, 0.05, 0.9),
        "lift_linear_velocity": ParameterPrior(0.2, 0.09, 0.04, 0.4),
        "object_friction": ParameterPrior(0.035, 0.01, 0.02, 0.05),
    }


def wide_place_priors() -> dict[str, ParameterPrior]:
    """
    :return: Priors of a first attempt, reaching past the range the successful
        attempts used.
    """
    return {
        "transport_linear_velocity": ParameterPrior(0.15, 0.07, 0.02, 0.3),
        "placing_linear_velocity": ParameterPrior(0.12, 0.06, 0.01, 0.25),
        "release_opening_velocity": ParameterPrior(0.15, 0.07, 0.015, 0.28),
        "retract_linear_velocity": ParameterPrior(0.17, 0.07, 0.02, 0.3),
    }


# %% what happened


@dataclass
class AttemptRecord:
    """One attempt at stacking a cube: the parameters it used, whether it
    stacked the cube and, after a failure, the diagnosis of both actions."""

    parameters: StackingParameters
    """The parameters the attempt used."""

    stacked: bool
    """Whether the cube ended up resting on the one it was stacked onto."""

    failure: str | None = None
    """Why the plan itself failed, if it did."""

    diagnoses: list[ActionDiagnosis] = field(default_factory=list)
    """The diagnosis of the pick-up and the place, after a failed attempt."""


@dataclass
class StepRecord:
    """Stacking one cube onto another, with every attempt it took."""

    cube: CubeName
    """The cube stacked."""

    support: CubeName
    """The cube it was stacked onto."""

    attempts: list[AttemptRecord] = field(default_factory=list)
    """The attempts, in order."""

    @property
    def label(self) -> str:
        """The step, for the log."""
        return f"{self.cube} onto {self.support}"

    @property
    def succeeded(self) -> bool:
        """Whether the last attempt stacked the cube."""
        return bool(self.attempts) and self.attempts[-1].stacked

    @property
    def recovered(self) -> bool:
        """Whether the step succeeded only after its first attempt failed."""
        return self.succeeded and len(self.attempts) > 1


# %% the demonstration


@dataclass
class SimulatedJointPositions(MeasuredJointPositions):
    """
    The positions MuJoCo simulates for a robot's actuated joints, whose world positions
    are the set points they are commanded to: fingers closed on a cube stop at its
    faces.
    """

    simulation: MujocoSim
    """
    The running simulation.
    """

    simulated_names: dict[str, str]
    """
    The name of the simulated joint of each actuated joint, by the joint's full name.
    """

    @classmethod
    def of_robot(
        cls, simulation: MujocoSim, robot: AbstractRobot
    ) -> SimulatedJointPositions:
        """
        :return: The simulated positions of every joint of ``robot`` with one degree of
            freedom that the simulation models.
        """
        simulator = simulation.simulator
        simulated_names = {}
        for connection in robot.connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            name = simulation.synchronizer.element_names.of(connection)
            if (
                simulator.get_joint(name).type
                == SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION
            ):
                simulated_names[str(connection.name)] = name
        return cls(simulation=simulation, simulated_names=simulated_names)

    def positions(self) -> dict[str, float]:
        values = self.simulation.simulator.get_joints_values(
            list(self.simulated_names.values())
        ).result
        return {
            name: float(values[simulated])
            for name, simulated in self.simulated_names.items()
        }


@dataclass
class DiagnosedStackingDemonstration(RobotDemonstration):
    """The Panda stacks every cube onto the one before it, diagnosing each
    failed attempt and retrying it with the perfect parameters."""

    used_robot: type[AbstractRobot] = Panda

    ros_node_name: str = "panda_ijcai_demo"

    executor_type: type[PlanExecutor] = PhysicallySimulatedPlanExecutor

    event_segmentation: bool = False

    iterations: int = 20
    """How often the tower is built, from the cubes standing in a row."""

    attempts_per_cube: int = 4
    """How many attempts stacking one cube takes at most, the first one
    included."""

    seed: int = 42
    """The seed of the first attempts' parameters."""

    live_port: int = 0
    """The port the cramera viewer follows the live run on; ``0`` shows nothing
    in cramera."""

    apartment: Path | None = (
        Path(__file__).parents[2] / "resources" / "worlds" / "apartment.urdf"
    )
    """The apartment cramera draws around the robot, which nothing simulates;
    ``None`` draws none and keeps the floor visible."""

    apartment_origin: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            -2.45, -2.7, -0.947
        )
    )
    """Where the apartment's root lies relative to the robot: the robot stands
    on the kitchen island, 16 cm from its edge, and the island's top is the
    floor."""

    report: Path = Path(__file__).parent / "inference_report.md"
    """The file every iteration's attempts and diagnoses are appended to."""

    cube_edge_length: float = 0.05
    """Edge length of every cube."""

    cube_mass: float = 0.01875
    """Mass of every cube: a 5 cm cube of density 150 kg/m^3."""

    cubes: list[CubeSpawn] = field(
        default_factory=lambda: [
            CubeSpawn(CubeName.CUBE_0, 0.40, 0.10, Color.from_rgba(0.9, 0.315, 0.315)),
            CubeSpawn(CubeName.CUBE_1, 0.40, -0.04, Color.from_rgba(0.315, 0.9, 0.315)),
            CubeSpawn(CubeName.CUBE_2, 0.40, -0.14, Color.from_rgba(0.315, 0.9, 0.315)),
            CubeSpawn(CubeName.CUBE_3, 0.40, -0.24, Color.from_rgba(0.315, 0.9, 0.315)),
        ]
    )
    """The cubes in stacking order; the first is the bottom of the tower and
    stays put."""

    grasp_height: float = 0.005
    """How far above its centre a cube is gripped, which keeps the fingertips
    clear of the cube below when it is let go of on the tower."""

    release_height: float = 0.003
    """How far above the cube below it a cube is let go of."""

    stack_horizontal_tolerance: float = 0.02
    """Greatest horizontal distance between a cube and the one below it that
    counts as stacked."""

    stack_height_tolerance: float = 0.012
    """Greatest deviation from one edge length above the cube below that counts
    as stacked."""

    settling_after_attempt: timedelta = timedelta(seconds=1.5)
    """Simulated time a cube is given to come to rest before an attempt is
    judged."""

    friction_cone: mujoco.mjtCone = mujoco.mjtCone.mjCONE_ELLIPTIC
    """The friction cone of the contacts: the exact elliptic one, where the
    default pyramid lets a squeezed cube creep between the fingers."""

    impedance_ratio: float = 10.0
    """How much harder sliding is than pressing in, MuJoCo's guidance for
    grasping with an elliptic cone."""

    integrator: mujoco.mjtIntegrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    """The numerical integrator of the physics: implicit in the velocity-
    dependent forces but without their Coriolis terms, a quarter of the
    dynamics evaluations per step of the simulator's default fourth-order
    Runge-Kutta."""

    pick_up_diagnoser: ActionCausalDiagnoser = field(
        default_factory=lambda: ActionCausalDiagnoser(DiagnosedAction.PICK_UP)
    )
    """Diagnoses the pick-up of a failed attempt."""

    place_diagnoser: ActionCausalDiagnoser = field(
        default_factory=lambda: ActionCausalDiagnoser(DiagnosedAction.PLACE)
    )
    """Diagnoses the place of a failed attempt."""

    generator: np.random.Generator = field(init=False)
    """Samples the first attempts' parameters."""

    segmentation: Segmind | None = field(default=None, init=False)
    """
    Segments what happens to the cubes into events while the run lasts.
    """

    live: LiveVisualization | None = field(default=None, init=False)
    """Serves the run to the cramera viewer, once the world is acquired."""

    tower: int = field(default=0, init=False)
    """The number of the tower being built."""

    diagnosed_steps: int = field(default=0, init=False)
    """How many steps were diagnosed so far."""

    planned: tuple[CubeName, CubeName, StackingParameters] | None = field(
        default=None, init=False
    )
    """The cube, the cube it is stacked onto and the parameters of the attempt
    :meth:`build_statechart` builds next."""

    def __post_init__(self) -> None:
        self.generator = np.random.default_rng(self.seed)

    # %% the scene

    def scene_specification(self) -> WorldSpecification:
        return WorldSpecification(
            objects=[
                BodySpecification.box(
                    "floor",
                    Scale(2.0, 2.0, 0.02),
                    parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01),
                )
            ]
        )

    def robot_specification(self) -> RobotSpecification:
        return RobotSpecification(self.used_robot)

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(CubeName.CUBE_0)

    def populate_scene(self, world: World) -> None:
        """Stand every cube in its place on the floor, free to be picked up."""
        inertia = self.cube_mass * 2 * self.cube_edge_length**2 / 12
        for spawn in self.cubes:
            annotation = SemanticAnnotationWithRootSpecification(
                name=spawn.name,
                semantic_annotation_type=HasGraspCandidates,
                root_specification=BodySpecification.box(
                    spawn.name,
                    Scale(
                        self.cube_edge_length,
                        self.cube_edge_length,
                        self.cube_edge_length,
                    ),
                    color=spawn.color,
                    connection_specification=Connection6DoFSpecification(),
                ),
            ).spawn(world)
            ContactParameters.create_for_grasped_object(sliding_friction=1.5).apply_to(
                [annotation.root]
            )
            annotation.root.inertial = Inertial(
                mass=self.cube_mass,
                inertia=InertiaTensor.from_values(inertia, inertia, inertia, 0, 0, 0),
            )
            annotation.root.parent_connection.origin = self._spawn_pose(spawn, world)
        if self.apartment is not None:
            floor = world.get_body_by_name("floor")
            with world.modify_world():
                floor.update_shapes(ShapeCollection(), floor.collision)

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [RobotAccess(world.get_semantic_annotations_by_type(Panda)[0])]

    def build_simulation(self, world: World) -> MujocoSim:
        return MujocoSim(
            world=world,
            headless=self.headless,
            cone=self.friction_cone,
            impratio=self.impedance_ratio,
            integrator=self.integrator,
        )

    def acquire_world(self) -> World:
        """Extends :meth:`RobotDemonstration.acquire_world` by serving the
        world to the cramera viewer with :meth:`build_backdrop` around it and
        the events segmind detects, unless :attr:`live_port` is ``0``."""
        world = super().acquire_world()
        if self.live_port:
            self.live = LiveVisualization(
                world=world,
                port=self.live_port,
                backdrop=self.build_backdrop(),
                detect_events=True,
            ).start()
            self.visualization = CrameraVisualization(live=self.live)
        return world

    def start_simulation(self, world: World) -> None:
        """
        Extends :meth:`RobotDemonstration.start_simulation` by showing in cramera where
        the robot's joints are in the simulation rather than their set points.
        """
        super().start_simulation(world)
        if self.live is not None and self.simulation is not None:
            self.live.publisher.state.measured = SimulatedJointPositions.of_robot(
                self.simulation, world.get_semantic_annotations_by_type(Panda)[0]
            )

    def started_segmentation(self, world: World) -> Segmind:
        """
        :return: Segmind watching the cubes: when the run is live, the one cramera
            shows in its Segmind tab, otherwise one of the demonstration's own.
        """
        if self.live is not None and self.live.detections.segmentation is not None:
            return self.live.detections.segmentation
        segmentation = Segmind.create_for_bodies(
            world, [world.get_body_by_name(cube.name) for cube in self.cubes]
        )
        segmentation.start()
        return segmentation

    def build_backdrop(self) -> World | None:
        """
        :return: What is drawn of the apartment, at :attr:`apartment_origin`,
            or ``None`` without :attr:`apartment`.
        """
        if self.apartment is None:
            return None
        apartment = URDFParser(urdf=VisualDescription(self.apartment).text())
        backdrop = WorldSpecification().create_world()
        backdrop.merge_world_at_pose(
            WorldSpecification(world_source=apartment).create_world(),
            self.apartment_origin,
        )
        return backdrop

    # %% one attempt

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """Park, pick up the cube of :attr:`planned`, carry it straight to the
        cube below, set it down there and park again."""
        cube_name, support_name, parameters = self.planned
        world = context.world
        panda = context.require_extension(RobotAccess).robot
        cube = self._annotation_of(cube_name, world)
        support = world.get_body_by_name(support_name)
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=panda.all_arms),
                    PickUpAction(
                        grasp=self._grasp_from_above(cube),
                        arm=panda.arm,
                        **parameters.pick_up_fields(),
                    ),
                    PlaceAction(
                        object_designator=cube,
                        target_location=self._place_pose_on(support, cube.root),
                        **parameters.place_fields(),
                    ),
                    ParkArmsAction(arms=panda.all_arms),
                ]
            )
        )
        return statechart

    def attempt(
        self,
        world: World,
        cube: CubeName,
        support: CubeName,
        parameters: StackingParameters,
    ) -> AttemptRecord:
        """Perform one attempt at stacking ``cube`` onto ``support`` with
        ``parameters``, and judge it once the cube came to rest."""
        self._set_friction(world.get_body_by_name(cube), parameters.friction)
        self.planned = (cube, support, parameters)
        executor = self.create_executor(world)
        failure = None
        try:
            executor.compile(self.build_statechart(executor.context))
            executor.execute()
        except (PlanFailure, MotionDidNotFinish) as plan_failure:
            failure = type(plan_failure).__name__
            self._park_and_open(world)
        finally:
            executor.close()
        self.simulation.step_simulation(self.settling_after_attempt)
        self._log_pose(world.get_body_by_name(cube))
        return AttemptRecord(
            parameters=parameters,
            stacked=self._is_stacked(world, cube, support),
            failure=failure,
        )

    def diagnose(self, cube: CubeName, attempt: AttemptRecord) -> None:
        """Diagnose the pick-up and the place of a failed attempt, against the
        final height of the cube it stacked."""
        attempt.diagnoses = [
            self.pick_up_diagnoser.diagnose(attempt.parameters.pick_up, cube),
            self.place_diagnoser.diagnose(attempt.parameters.place, cube),
        ]

    def stack(
        self,
        world: World,
        cube: CubeName,
        support: CubeName,
        first_parameters: StackingParameters,
    ) -> StepRecord:
        """Stack ``cube`` onto ``support``: first with ``first_parameters``,
        and after a failure, which is diagnosed, with the perfect parameters
        until it stands or :attr:`attempts_per_cube` attempts were made."""
        step = StepRecord(cube=cube, support=support)
        parameters = first_parameters
        diagnosed: DiagnosedStep | None = None
        while len(step.attempts) < self.attempts_per_cube:
            started = datetime.now()
            attempt = self.attempt(world, cube, support, parameters)
            step.attempts.append(attempt)
            self._log_attempt(step, attempt)
            if diagnosed is None and not attempt.stacked:
                self.diagnose(cube, attempt)
                for diagnosis in attempt.diagnoses:
                    print(
                        f"[causal] {step.label} ({diagnosis.action.name.lower()}): {diagnosis.explanation()}"
                    )
                diagnosed = self._diagnosed_step(world, step, started)
                print(f"[failure] {step.label}: {diagnosed.failure}")
                for observation in diagnosed.observations:
                    print(f"[segmind] {step.label}: {observation}")
            if diagnosed is not None:
                self._publish_recovery(diagnosed, step)
            if attempt.stacked:
                return step
            parameters = self.perfect_parameters(step.attempts[0])
        return step

    def perfect_parameters(self, failed_attempt: AttemptRecord) -> StackingParameters:
        """
        :param failed_attempt: The diagnosed first attempt of a step.
        :return: The perfect value of every parameter of both actions.
        """
        pick_up, place = failed_attempt.diagnoses
        return StackingParameters(
            pick_up=pick_up.perfect_parameters, place=place.perfect_parameters
        )

    def sampled_parameters(self) -> StackingParameters:
        """
        :return: Parameters of a first attempt, sampled from the wide priors.
        """
        return StackingParameters(
            pick_up={
                name: prior.sample(self.generator)
                for name, prior in wide_pick_up_priors().items()
            },
            place={
                name: prior.sample(self.generator)
                for name, prior in wide_place_priors().items()
            },
        )

    # %% the run

    def run(self) -> World:
        """Build the tower :attr:`iterations` times, from the cubes standing in
        a row, and report how every step went."""
        try:
            world = self.acquire_world()
            self.start_simulation(world)
            self.segmentation = self.started_segmentation(world)
            built = 0
            for iteration in range(1, self.iterations + 1):
                started = perf_counter()
                print(f"=== iteration {iteration}/{self.iterations} ===")
                self.tower = iteration
                self._reset_scene(world)
                steps = self._build_tower(world)
                standing = self._tower_stands(world)
                built += standing
                self._append_report(iteration, steps, standing)
                print(
                    f"=== iteration {iteration} {'built the tower' if standing else 'did NOT build the tower'}"
                    f" in {perf_counter() - started:.0f} s ==="
                )
            print(f"[summary] {built}/{self.iterations} towers built")
        finally:
            if self.segmentation is not None:
                self.segmentation.stop()
            self.tear_down()
        return world

    def _build_tower(self, world: World) -> list[StepRecord]:
        """Stack every cube onto the one before it, re-stacking a lower cube an
        attempt knocked off, and stop at the first cube that does not stand."""
        steps = []
        stacked: list[tuple[CubeName, CubeName]] = []
        for support, cube in zip(self.cubes, self.cubes[1:]):
            step = self.stack(world, cube.name, support.name, self.sampled_parameters())
            steps.append(step)
            if not step.succeeded:
                return steps
            for lower_cube, lower_support in stacked:
                if self._is_stacked(world, lower_cube, lower_support):
                    continue
                print(
                    f"[warning] {lower_cube} was knocked off {lower_support}, re-stacking it"
                )
                restack = self.stack(
                    world,
                    lower_cube,
                    lower_support,
                    self.perfect_parameters(self._diagnosed_first_attempt(steps)),
                )
                steps.append(restack)
                if not restack.succeeded:
                    return steps
            stacked.append((cube.name, support.name))
        return steps

    def _diagnosed_first_attempt(self, steps: list[StepRecord]) -> AttemptRecord:
        """
        :return: A first attempt of ``steps`` that was diagnosed, or the diagnosis of
            the last first attempt made now, for the perfect parameters it names.
        """
        for step in steps:
            if step.attempts[0].diagnoses:
                return step.attempts[0]
        first = steps[-1].attempts[0]
        self.diagnose(steps[-1].cube, first)
        return first

    # %% resetting the scene

    def _reset_scene(self, world: World) -> None:
        """Stand every cube back in its place, bring the arm to its park
        position with the hand open, and let the scene come to rest."""
        for spawn in self.cubes:
            world.get_body_by_name(
                spawn.name
            ).parent_connection.origin = self._spawn_pose(spawn, world)
            self._set_friction(world.get_body_by_name(spawn.name), 1.5)
        self._park_and_open(world)

    def _park_and_open(self, world: World) -> None:
        """Put the arm in its park position with the hand open, at rest,
        without a plan:

        the joints and their servos' set points jump there.
        """
        panda = world.get_semantic_annotations_by_type(Panda)[0]
        states = [
            panda.arm.get_joint_state_by_type(StaticJointState.PARK),
            panda.arm.end_effector.get_joint_state_by_type(GripperState.OPEN),
        ]
        for joint_state in states:
            for connection, value in zip(
                joint_state.connections, joint_state.target_values
            ):
                world.state[connection.raw_dof.id].position = value
        world.notify_state_change()
        self.simulation.simulator._mj_data.qvel[:] = 0
        self.simulation.synchronizer.command_actuators_from_world_state()
        self.simulation.step_simulation(self.settling_duration)

    # %% the geometry of the tower

    def _spawn_pose(
        self, spawn: CubeSpawn, world: World
    ) -> HomogeneousTransformationMatrix:
        """
        :return: Where ``spawn``'s cube stands on the floor.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            spawn.x, spawn.y, self.cube_edge_length / 2, reference_frame=world.root
        )

    def _annotation_of(self, cube: CubeName, world: World) -> HasGraspCandidates:
        """
        :return: The annotation of ``cube`` in ``world``.
        """
        body = world.get_body_by_name(cube)
        [annotation] = [
            annotation
            for annotation in world.get_semantic_annotations_by_type(HasGraspCandidates)
            if annotation.root is body
        ]
        return annotation

    def _grasp_from_above(self, cube: HasGraspCandidates) -> GraspCandidate:
        """
        :return: The grasp coming straight down onto ``cube`` wherever it lies, turned
            about the vertical to meet the faces of the cube, :attr:`grasp_height`
            above its centre.
        """
        world = cube.root._world
        position = cube.root.global_pose.position.to_np()[:3]
        down = RotationMatrix.from_vectors(x=Vector3.NEGATIVE_Z(), y=Vector3.X())
        turned = RotationMatrix.from_rpy(yaw=self._face_yaw(cube.root)) @ down
        world_grasp = Pose(
            position=Point3(
                position[0],
                position[1],
                position[2] + self.grasp_height,
                reference_frame=world.root,
            ),
            orientation=turned.quaternion,
            reference_frame=world.root,
        )
        return GraspCandidate(cube, world.transform(world_grasp, cube.root))

    @staticmethod
    def _face_yaw(cube: Body) -> float:
        """
        :return: The yaw of ``cube``'s horizontal axes, whichever face points up, folded
            into [-45, 45) degrees, since a cube looks the same every quarter turn.
        """
        rotation = cube.global_pose.rotation_matrix.to_np()[:3, :3]
        upward_axis = int(np.argmax(np.abs(rotation[2, :3])))
        horizontal_axis = rotation[:2, (upward_axis + 1) % 3]
        yaw = float(np.arctan2(horizontal_axis[1], horizontal_axis[0]))
        return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _place_pose_on(self, support: Body, cube: Body) -> Pose:
        """
        :return: Where ``cube`` is let go of: straight above where ``support`` stands
            now, :attr:`release_height` above it, turned as ``cube`` is now, so the hand
            holding it need not turn.
        """
        world = support._world
        below = support.global_pose.position.to_np()[:3]
        orientation = cube.global_pose.rotation_matrix.quaternion
        return Pose(
            position=Point3(
                below[0],
                below[1],
                below[2] + self.cube_edge_length + self.release_height,
                reference_frame=world.root,
            ),
            orientation=orientation,
            reference_frame=world.root,
        )

    def _is_stacked(self, world: World, cube: CubeName, support: CubeName) -> bool:
        """
        :return: Whether ``cube`` rests squarely on ``support``: one edge length above
            it and centred on it, within the tolerances.
        """
        placed = world.get_body_by_name(cube).global_pose.position.to_np()[:3]
        below = world.get_body_by_name(support).global_pose.position.to_np()[:3]
        return (
            np.linalg.norm(placed[:2] - below[:2]) <= self.stack_horizontal_tolerance
            and abs(placed[2] - below[2] - self.cube_edge_length)
            <= self.stack_height_tolerance
        )

    def _tower_stands(self, world: World) -> bool:
        """
        :return: Whether every cube rests on the one before it.
        """
        return all(
            self._is_stacked(world, cube.name, support.name)
            for support, cube in zip(self.cubes, self.cubes[1:])
        )

    def _set_friction(self, cube: Body, friction: float) -> None:
        """Give every geom of ``cube`` the sliding friction ``friction`` in the
        physics."""
        model = self.simulation.simulator._mj_model
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.simulation.synchronizer.element_names.of(cube),
        )
        first = model.body_geomadr[body_id]
        for geom in range(first, first + model.body_geomnum[body_id]):
            model.geom_friction[geom][0] = friction
            model.geom_priority[geom] = 1

    # %% reporting

    def _log_pose(self, cube: Body) -> None:
        """Print where ``cube`` lies: its position, the yaw of its faces and
        how far it is tilted off any face, in degrees."""
        position = cube.global_pose.position.to_np()[:3]
        rotation = cube.global_pose.rotation_matrix.to_np()[:3, :3]
        tilt = np.degrees(np.arccos(np.clip(np.max(np.abs(rotation[2, :3])), -1, 1)))
        print(
            f"[pose] {cube.name.name} at ({position[0]:.3f}, {position[1]:.3f}, "
            f"{position[2]:.3f}), yaw {np.degrees(self._face_yaw(cube)):.0f}, "
            f"tilt {tilt:.0f}"
        )

    # %% the diagnoses shown in cramera

    def _diagnosed_step(
        self, world: World, step: StepRecord, started: datetime
    ) -> DiagnosedStep:
        """
        :param started: When the diagnosed first attempt of ``step`` began.
        :return: What went wrong in that attempt, what segmind observed during it and
            every parameter its diagnoses judged.
        """
        self.diagnosed_steps += 1
        attempt = step.attempts[0]
        return DiagnosedStep(
            identifier=str(self.diagnosed_steps),
            title=f"Tower {self.tower}: {step.label}",
            failure=self._failure_of(world, step, attempt),
            observations=self._observations(world, step.cube, started),
            corrections=[
                ParameterCorrection(
                    action=diagnosis.action.name.lower().replace("_", " "),
                    parameter=judgement.parameter_name,
                    failed_value=judgement.observed_value,
                    failed_support=judgement.observed_support,
                    corrected_value=judgement.perfect_value,
                    corrected_support=judgement.perfect_support,
                    verdict=(
                        ParameterVerdict.PRIMARY_CAUSE
                        if judgement is diagnosis.anomalous[0]
                        else (
                            ParameterVerdict.ANOMALOUS
                            if judgement in diagnosis.anomalous
                            else ParameterVerdict.PLAUSIBLE
                        )
                    ),
                )
                for diagnosis in attempt.diagnoses
                for judgement in diagnosis.judgements
            ],
        )

    def _failure_of(
        self, world: World, step: StepRecord, attempt: AttemptRecord
    ) -> str:
        """
        :return: Where the cube of ``step`` came to rest after ``attempt``, as a
            sentence.
        """
        position = world.get_body_by_name(step.cube).global_pose.position.to_np()[:3]
        if position[2] >= self.cube_edge_length:
            sentence = f"{step.cube} did not come to rest on {step.support}"
        else:
            [spawn] = [spawn for spawn in self.cubes if spawn.name == step.cube]
            moved = np.linalg.norm(position[:2] - np.array([spawn.x, spawn.y]))
            sentence = (
                f"{step.cube} slid out of the fingers and stayed where it stood"
                if moved <= self.stack_horizontal_tolerance
                else f"{step.cube} slipped out of the fingers and fell"
            )
        if attempt.failure:
            sentence += f"; the plan failed with {attempt.failure}"
        return sentence + "."

    def _observations(
        self, world: World, cube: CubeName, started: datetime
    ) -> list[str]:
        """
        :return: Every event segmind detected about ``cube`` since ``started``, each as
            its time since then and its words.
        """
        if self.segmentation is None:
            return []
        body = world.get_body_by_name(cube)
        observations = []
        for event in self.segmentation.event_logger.get_events():
            if event.timestamp < started or body not in event.tracked_objects:
                continue
            words = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])",
                " ",
                type(event).__name__.removesuffix("Event"),
            ).capitalize()
            subject = f"{words}: {event.tracked_object.name.name}"
            if event.with_object is not None:
                subject += f" with {event.with_object.name.name}"
            elapsed = (event.timestamp - started).total_seconds()
            observations.append(f"+{elapsed:.1f} s  {subject}")
        return observations

    def _publish_recovery(self, diagnosed: DiagnosedStep, step: StepRecord) -> None:
        """Report how far the recovery of ``step`` has come, on the console and
        in cramera."""
        diagnosed.retries = len(step.attempts) - 1
        if step.succeeded:
            diagnosed.status = DiagnosisStatus.RECOVERED
            retries = "retry" if diagnosed.retries == 1 else "retries"
            print(
                f"[recovery] {step.label} recovered after {diagnosed.retries} {retries}"
                " with the corrected parameters"
            )
        elif len(step.attempts) >= self.attempts_per_cube:
            diagnosed.status = DiagnosisStatus.NOT_RECOVERED
            print(f"[recovery] {step.label} not recovered")
        if self.live is not None:
            self.live.publisher.diagnoses.publish(replace(diagnosed))

    def _log_attempt(self, step: StepRecord, attempt: AttemptRecord) -> None:
        """Print one attempt: its parameters and how it went."""
        number = len(step.attempts)
        kind = (
            "first attempt" if number == 1 else f"attempt {number} (perfect parameters)"
        )
        print(f"[params] {step.label} {kind}: {attempt.parameters.described()}")
        failure = f", plan failed: {attempt.failure}" if attempt.failure else ""
        print(
            f"[info] {step.label} {kind} {'is stacked' if attempt.stacked else 'is NOT stacked'}{failure}"
        )

    def _append_report(
        self, iteration: int, steps: list[StepRecord], standing: bool
    ) -> None:
        """Append how every step of ``iteration`` went, with its diagnoses, to
        :attr:`report`."""
        lines = [f"\n## Iteration {iteration}\n", f"Tower built: **{standing}**\n"]
        for step in steps:
            lines.append(f"\n### {step.label}\n")
            for number, attempt in enumerate(step.attempts, start=1):
                outcome = "stacked" if attempt.stacked else "not stacked"
                failure = (
                    f" (plan failed: {attempt.failure})" if attempt.failure else ""
                )
                lines.append(f"- attempt {number}: {outcome}{failure}")
                for diagnosis in attempt.diagnoses:
                    lines.append(
                        f"  - {diagnosis.action.name.lower()}: {diagnosis.explanation()}"
                    )
            lines.append(
                f"- result: {'recovered' if step.recovered else 'SUCCESS' if step.succeeded else 'FAILURE'}"
            )
        with self.report.open("a") as report:
            report.write("\n".join(lines) + "\n")


def main() -> None:
    """Run the demonstration from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--live-port", type=int, default=0)
    arguments = parser.parse_args()
    DiagnosedStackingDemonstration(
        iterations=arguments.iterations,
        seed=arguments.seed,
        headless=arguments.headless,
        live_port=arguments.live_port,
    ).run()


if __name__ == "__main__":
    main()
