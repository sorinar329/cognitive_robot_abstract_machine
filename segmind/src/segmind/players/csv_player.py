import logging
import threading
from dataclasses import dataclass, field
import pandas as pd

from segmind.players.data_player import FilePlayer, FrameData
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, Set

logger = logging.getLogger(__name__)

EXCLUDED_KEYWORDS = {"velocity", "actuator"}


@dataclass(eq=False)
class CSVEpisodePlayer(FilePlayer):
    """
    Plays the episode from a CSV file.
    """

    data_frames: pd.DataFrame = field(default=None, init=False, hash=False, compare=False)
    """
    The data frames of the CSV file.
    """

    data_object_names: Set[str] = field(default=None, init=False, hash=False, compare=False)
    """
    The names of the objects in the CSV file.
    """

    def __post_init__(self):
        super().__post_init__()



    def get_frame_data_generator(self):
        """
        Reads the CSV file and generates the frame data.
        """
        logger.debug(f"Reading CSV file {self.file_path}")
        self.data_frames = pd.read_csv(self.file_path)

        def _is_excluded(name: str) -> bool:
            lower = name.lower()
            return any(keyword in lower for keyword in EXCLUDED_KEYWORDS)

        self.data_object_names = {
            v.split(":")[0] for v in self.data_frames.columns
            if ":" in v and not _is_excluded(v.split(":")[0])
        }

        excluded_cols = {
            col for col in self.data_frames.columns
            if ":" in col and _is_excluded(col.split(":")[0])
        }
        filtered_frames = self.data_frames.drop(columns=excluded_cols)

        for i, (frame_id, objects_data) in enumerate(filtered_frames.iterrows()):
            yield FrameData(
                time=float(objects_data["time"]),
                objects_data=objects_data.to_dict(),
                frame_idx=i,
            )

    def _pause(self):
        """
        Pauses the episode player.
        """

    def _resume(self):
        """
        Resumes the episode player.
        """

    def get_objects_poses(self, frame_data: FrameData) -> Dict[Body, Pose]:
        """
        Extracts the poses of the objects from the frame data.

        :param frame_data: The frame data.
        :return: A dictionary mapping bodies to poses.
        """
        objects_poses: Dict[Body, Pose] = {}
        objects_data = frame_data.objects_data
        current_time = frame_data.time

        for obj_name in self.data_object_names:
            if "joint" in obj_name:
                continue

            # Store raw floats, no Pose object created
            objects_poses[obj_name] = (
                objects_data[f"{obj_name}:position_0"],
                objects_data[f"{obj_name}:position_1"],
                objects_data[f"{obj_name}:position_2"],
                objects_data[f"{obj_name}:quaternion_0"],  # w
                objects_data[f"{obj_name}:quaternion_1"],  # x
                objects_data[f"{obj_name}:quaternion_2"],  # y
                objects_data[f"{obj_name}:quaternion_3"],  # z
            )
        return objects_poses


    def get_joint_states(self, frame_data: FrameData) -> Dict[str, float]:
        """
        Extracts the joint states from the frame data.

        :param frame_data: The frame data.
        :return: A dictionary mapping joint names to their positions.
        """
        joint_states = {}
        for col in self.data_frames.columns:
            if ":" in col and ("joint_angular_position" in col.lower() or "joint_linear_position" in col.lower()) and "cmd" not in col.lower():
                joint_name = col.split(":")[0]
                joint_position = frame_data.objects_data[col]
                joint_states[joint_name] = joint_position
        return joint_states