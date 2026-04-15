import os
from dataclasses import dataclass, field

import logging
from pathlib import Path
import mujoco
from xml.etree import ElementTree as ET

from semantic_digital_twin.adapters.package_resolver import FileUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

try:
    import sapien
except ImportError:
    logger.warn(
        "Sapien library is required for Partnet Mobility dataset loading. "
        "Please install it using 'pip install -U 'sapien>=3.0.0b1'"
    )

SAPIEN_ACCESS_TOKEN_ENVIRONMENT_VARIABLE_NAME = "SAPIEN_ACCESS_TOKEN"


@dataclass
class PartnetMobilityDatasetLoader:
    """
    Loader for articulated assets from the partnet mobility dataset (https://sapien.ucsd.edu/browse).

    For this to work out of the box, the environment variable SAPIEN_ACCESS_TOKEN must be set
    and you have to install sapien.
    """

    token: str = field(
        default_factory=lambda: os.environ[
            SAPIEN_ACCESS_TOKEN_ENVIRONMENT_VARIABLE_NAME
        ]
    )
    """
    The token to use for communication with the partnet server.
    """

    directory: Path = field(
        default_factory=lambda: Path.home() / "partnet-mobility-dataset"
    )
    """
    The directory where to save the downloaded URDF files into.
    """

    def load(self, model_id: int = 179) -> World:
        """
        Load a world given the model id.

        :param model_id: The id of the model to load.
        :return: The loaded world.
        """
        urdf_file = sapien.asset.download_partnet_mobility(
            model_id=model_id, token=self.token, directory=self.directory
        )
        mj_root = ET.parse(urdf_file).getroot()
        # mujoco_element = ET.SubElement(mj_root, "mujoco")
        # compiler_element = ET.SubElement(mujoco_element, "compiler")
        # compiler_element.set("angle", "radian")
        # compiler_element.set("meshdir", "textured_objs")
        mj_string = ET.tostring(mj_root, encoding="unicode")
        mj_spec = mujoco.MjSpec.from_string(mj_string)
        print(mj_spec.to_xml())
        world = URDFParser.from_file(file_path=urdf_file).parse()
        return world

    def _add_effort_to_limit_tags(self, file_path: str, effort: float = 100.0):
