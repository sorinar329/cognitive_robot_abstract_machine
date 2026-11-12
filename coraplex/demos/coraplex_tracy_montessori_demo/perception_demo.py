"""
Perceives where the Montessori pieces stand on Tracy's table and moves them there in the
world, without running any plan.

Run it from this folder, with Giskard and the camera running::

    python perception_demo.py

The scene is stood in the world fetched from Giskard as :mod:`real_demo` stands it. One
colour image is then taken, every piece is found by its colour and outline, and the ray
through its middle is met with the plane of the pieces' top faces. Each piece's body is
moved to the x and y found that way; its height stays the one the scene gave it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from transforms3d.quaternions import quat2mat

from demo import MontessoriScene
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from montessori_board import HoleShape
from real_demo import RealRunner
from semantic_digital_twin.world import World

# %% one picture of the table


@dataclass
class CameraFrame:
    """
    One colour image of the table, with what is needed to cast rays through its pixels.
    """

    colour: np.ndarray
    """
    The image, in OpenCV's BGR order.
    """

    intrinsics: np.ndarray
    """
    The 3x3 pinhole camera matrix the image was taken with.
    """

    distortion: np.ndarray
    """
    The lens distortion coefficients the image was taken with.
    """

    world_T_camera: np.ndarray
    """
    The 4x4 pose of the camera's optical frame in the world's root frame.
    """

    def ray_through(self, pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """
        :param pixel: A pixel of the image, as column and row.
        :return: Where the ray through that pixel starts and the direction it runs in,
            both in the world's root frame.
        """
        [[normalised]] = cv2.undistortPoints(
            np.array([[pixel]], dtype=np.float64), self.intrinsics, self.distortion
        )
        camera_direction = np.array([normalised[0], normalised[1], 1.0])
        rotation = self.world_T_camera[:3, :3]
        return self.world_T_camera[:3, 3], rotation @ camera_direction

    def point_on_plane(
        self, pixel: tuple[float, float], plane_height: float
    ) -> np.ndarray:
        """
        :param pixel: A pixel of the image, as column and row.
        :param plane_height: The height of a horizontal plane in the world's root frame.
        :return: Where the ray through the pixel meets that plane.
        """
        origin, direction = self.ray_through(pixel)
        distance = (plane_height - origin[2]) / direction[2]
        return origin + distance * direction


@dataclass
class TableCamera:
    """
    The camera looking down at Tracy's table, read through ROS.
    """

    node: Node
    """
    The node the subscriptions and the transform listener live on; it has to be spun.
    """

    colour_topic: str = "/camera/color/image_raw"
    """
    The topic the colour images are published on.
    """

    camera_info_topic: str = "/camera/color/camera_info"
    """
    The topic the colour camera's calibration is published on.
    """

    world_frame: str = "map"
    """
    The TF frame that coincides with the root of the world fetched from Giskard.
    """

    wait_time: timedelta = timedelta(seconds=10)
    """
    How long to wait for an image, its calibration and the camera's pose.
    """

    _transforms: Buffer = field(init=False)
    """
    The transforms received so far.
    """

    _transform_listener: TransformListener = field(init=False)
    """
    Fills :attr:`_transforms`; kept so it is not garbage collected.
    """

    def __post_init__(self) -> None:
        self._transforms = Buffer()
        self._transform_listener = TransformListener(self._transforms, self.node)

    def take_frame(self) -> CameraFrame:
        """
        :return: The next colour image published, with its calibration and the camera's
            pose at that moment.
        """
        image = self._next_message(Image, self.colour_topic)
        calibration = self._next_message(CameraInfo, self.camera_info_topic)
        transform = self._transforms.lookup_transform(
            self.world_frame,
            image.header.frame_id,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=self.wait_time.total_seconds()),
        ).transform
        world_T_camera = np.eye(4)
        world_T_camera[:3, :3] = quat2mat(
            [
                transform.rotation.w,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
            ]
        )
        world_T_camera[:3, 3] = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
        return CameraFrame(
            colour=self._bgr_image_of(image),
            intrinsics=np.array(calibration.k, dtype=np.float64).reshape(3, 3),
            distortion=np.array(calibration.d, dtype=np.float64),
            world_T_camera=world_T_camera,
        )

    def _next_message[MessageType](
        self, message_type: type[MessageType], topic: str
    ) -> MessageType:
        """
        :param message_type: The type of message the topic carries.
        :param topic: The topic to listen to.
        :return: The next message published on the topic.
        :raises TimeoutError: If none arrives within :attr:`wait_time`.
        """
        received: list[MessageType] = []
        arrived = threading.Event()

        def keep(message: MessageType) -> None:
            received.append(message)
            arrived.set()

        subscription = self.node.create_subscription(
            message_type, topic, keep, qos_profile_sensor_data
        )
        arrived.wait(self.wait_time.total_seconds())
        self.node.destroy_subscription(subscription)
        if not received:
            raise TimeoutError(f"Nothing was published on {topic}.")
        return received[0]

    @staticmethod
    def _bgr_image_of(image: Image) -> np.ndarray:
        """
        Converts by hand, since ``cv_bridge`` is built against an older NumPy.

        :param image: A colour image message in ``rgb8`` or ``bgr8``.
        :return: The image in OpenCV's BGR order.
        """
        pixels = np.frombuffer(image.data, np.uint8).reshape(
            image.height, image.width, 3
        )
        if image.encoding == "rgb8":
            return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        return pixels.copy()


# %% finding the pieces by their colours


@dataclass
class ColourBlob:
    """
    A patch of one piece colour in the image.
    """

    outline: np.ndarray
    """
    The patch's outer contour, in pixels.
    """

    @property
    def middle(self) -> tuple[float, float]:
        """
        :return: The patch's centre of area, as column and row.
        """
        moments = cv2.moments(self.outline)
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]

    @property
    def area(self) -> float:
        """
        :return: How many pixels the patch covers.
        """
        return cv2.contourArea(self.outline)

    @property
    def elongation(self) -> float:
        """
        :return: How much longer than wide the smallest rectangle around the patch is.
        """
        _, (width, height), _ = cv2.minAreaRect(self.outline)
        return max(width, height) / min(width, height)

    @property
    def fullness(self) -> float:
        """
        :return: How much of the smallest rectangle around the patch it fills.
        """
        _, (width, height), _ = cv2.minAreaRect(self.outline)
        return self.area / (width * height)


@dataclass
class PieceColour:
    """
    The colour two of the pieces share, and how to tell those two apart.
    """

    lowest_hue: int
    """
    The lowest OpenCV hue (0 to 180) the colour covers.
    """

    highest_hue: int
    """
    The highest OpenCV hue (0 to 180) the colour covers.
    """

    pieces: tuple[HoleShape, HoleShape]
    """
    The two pieces of this colour: the one that ranks higher by
    :meth:`rank` first.
    """

    ranks_by_elongation: bool
    """
    Whether the first of :attr:`pieces` is the more elongated patch, rather than the one
    filling its bounding rectangle more.
    """

    def rank(self, blob: ColourBlob) -> float:
        """
        :param blob: A patch of this colour.
        :return: How strongly the patch looks like the first of :attr:`pieces`.
        """
        return blob.elongation if self.ranks_by_elongation else blob.fullness


@dataclass
class PieceFinder:
    """
    Finds the four pieces in a colour image: two of them yellow, two of them cyan.
    """

    colours: list[PieceColour] = field(
        default_factory=lambda: [
            PieceColour(
                lowest_hue=15,
                highest_hue=35,
                pieces=(HoleShape.RECTANGLE, HoleShape.TRIANGLE),
                ranks_by_elongation=True,
            ),
            PieceColour(
                lowest_hue=85,
                highest_hue=110,
                pieces=(HoleShape.SQUARE, HoleShape.CIRCLE),
                ranks_by_elongation=False,
            ),
        ]
    )
    """
    The colours of the pieces; measured off the real set under the lab's light.
    """

    minimum_saturation: int = 120
    """
    How saturated a pixel has to be to belong to a piece; the table and the board are
    far greyer.
    """

    minimum_brightness: int = 80
    """
    How bright a pixel has to be to belong to a piece.
    """

    minimum_area: float = 350.0
    """
    How many pixels a patch has to cover to be a piece; the pastel drawer fronts of the
    board leave smaller ones.
    """

    def find(self, image: np.ndarray) -> dict[HoleShape, ColourBlob]:
        """
        :param image: A colour image of the table, in BGR order.
        :return: The patch each piece was found as; a piece missing from the image is
            missing here too.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        found: dict[HoleShape, ColourBlob] = {}
        for colour in self.colours:
            blobs = sorted(
                self._blobs_of(hsv, colour), key=colour.rank, reverse=True
            )[: len(colour.pieces)]
            found.update(zip(colour.pieces, blobs))
        return found

    def _blobs_of(self, hsv: np.ndarray, colour: PieceColour) -> list[ColourBlob]:
        """
        :param hsv: The image, in OpenCV's HSV.
        :param colour: The colour to look for.
        :return: The two largest patches of that colour big enough to be pieces.
        """
        mask = (
            (hsv[..., 0] >= colour.lowest_hue)
            & (hsv[..., 0] <= colour.highest_hue)
            & (hsv[..., 1] >= self.minimum_saturation)
            & (hsv[..., 2] >= self.minimum_brightness)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        outlines, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blobs = [ColourBlob(outline) for outline in outlines]
        blobs = [blob for blob in blobs if blob.area >= self.minimum_area]
        return sorted(blobs, key=lambda blob: blob.area, reverse=True)[
            : len(colour.pieces)
        ]


# %% placing the pieces where they were seen


@dataclass
class PerceivedPiece:
    """
    Where one piece was seen, next to where the scene had stood it.
    """

    hole: HoleShape
    """
    The hole the piece goes through, which names it.
    """

    pixel: tuple[float, float]
    """
    The middle of the patch the piece was found as.
    """

    position: np.ndarray
    """
    Where the piece was seen, in the world's root frame.
    """


@dataclass
class PerceptionRunner(RealRunner):
    """
    Stands the scene in the world fetched from Giskard, then moves every piece to where
    the camera sees it.
    """

    finder: PieceFinder = field(default_factory=PieceFinder)
    """
    Finds the pieces in the image.
    """

    marked_image_path: Path = Path("perception_seen.png")
    """
    Where the image is written with every piece found marked on it.
    """

    node_name: str = "tracy_montessori_perception_demo"
    """
    The name of the demo's own ROS node.
    """

    def run(self) -> None:
        with self.running_robot() as robot:
            self.scene.populate(robot.world)
            frame = TableCamera(node=robot.node).take_frame()
            seen = self.perceive(frame)
            self.mark(frame, seen)
            self.place(seen, robot.world)
            self.report(seen)
            input("Pieces placed where they were seen; check RViz, then press Enter.")

    def perceive(self, frame: CameraFrame) -> list[PerceivedPiece]:
        """
        :param frame: The picture of the table.
        :return: Every piece found in it, with where it stands: on the plane of the
            pieces' top faces, since the pieces stand on the table.
        """
        top_face_height = self.scene.table_top_z + self.scene.pieces.piece_height
        return [
            PerceivedPiece(
                hole=hole,
                pixel=blob.middle,
                position=frame.point_on_plane(blob.middle, top_face_height),
            )
            for hole, blob in self.finder.find(frame.colour).items()
        ]

    def place(self, seen: list[PerceivedPiece], world: World) -> None:
        """
        Move each piece's body to the x and y it was seen at, keeping its height.

        :param seen: The pieces the camera found.
        :param world: The world the scene stands in.
        """
        position_of = {piece.hole: piece.position for piece in seen}
        for piece, hole in self.scene.holes_of_pieces.items():
            if hole not in position_of:
                continue
            x, y, _ = position_of[hole]
            height = world.compute_forward_kinematics_np(world.root, piece.root)[2, 3]
            piece.root.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    x, y, height, reference_frame=world.root
                )
            )

    def mark(self, frame: CameraFrame, seen: list[PerceivedPiece]) -> None:
        """
        Write the image with every piece found marked and named.

        :param frame: The picture of the table.
        :param seen: The pieces found in it.
        """
        marked = frame.colour.copy()
        for index, piece in enumerate(sorted(seen, key=lambda piece: piece.pixel[0])):
            column, row = (int(round(value)) for value in piece.pixel)
            cv2.circle(marked, (column, row), 6, (0, 0, 255), -1)
            cv2.putText(
                marked,
                f"{piece.hole} ({piece.position[0]:.3f}, {piece.position[1]:.3f})",
                (column - 60, row - 20 if index % 2 == 0 else row + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(str(self.marked_image_path), marked)

    def report(self, seen: list[PerceivedPiece]) -> None:
        """
        Print where every piece was stood by the scene and where it was seen.

        :param seen: The pieces the camera found.
        """
        position_of = {piece.hole: piece.position for piece in seen}
        for index, hole in enumerate(self.scene.holes_of_pieces.values()):
            stood_y = (
                self.scene.piece_row_start_y + index * self.scene.piece_row_spacing
            )
            stood = f"({self.scene.piece_row_x:.3f}, {stood_y:.3f})"
            if hole not in position_of:
                print(f"{hole:>9}: stood at {stood}, not seen")
                continue
            x, y, _ = position_of[hole]
            print(f"{hole:>9}: stood at {stood}, seen at ({x:.3f}, {y:.3f})")
        print(f"Marked image written to {self.marked_image_path.resolve()}")


# %% entry point


if __name__ == "__main__":
    PerceptionRunner(
        scene=MontessoriScene(pieces_raised_by=0.04, board_raised_by=0.04)
    ).run()
