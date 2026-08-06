"""
Bundle a Gazebo/SDF world or model into the same self-contained URDF format
:mod:`cram_viz.onboard.bundle_urdf` produces, so the web viewer renders it with the
identical pipeline it already knows how to load.

Where :func:`cram_viz.onboard.bundle_urdf.bundle_urdf` rewrites the mesh references of
an existing URDF file, this module builds a URDF from scratch: :class:`GazeboParser` has
already resolved every shape and pose into a
:class:`~semantic_digital_twin.world.World`, so bundling only has to hand it to
:func:`cram_viz.onboard.world_to_urdf.write_world_as_urdf`.
"""

from __future__ import annotations

import os

from typing_extensions import Any, Dict, Optional

from cram_viz.onboard.bundle_urdf import resolve_uri
from cram_viz.onboard.world_to_urdf import write_world_as_urdf
from semantic_digital_twin.adapters.gazebo import GazeboParser

#: directory a bundled mesh's own source directory name is nested under, so meshes from
#: differently named Gazebo models cannot collide
GAZEBO_MESH_DIRECTORY = "gazebo"


# %% bundling
def bundle_gazebo_world(
    source: str, name: str, out_dir: str, hints: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Bundle one Gazebo/SDF world or model, with every mesh it references, into a self-
    contained URDF.

    :param source: Path or URI of the world/model file to bundle.
    :param name: Output model name, used for ``<out_dir>/<name>.urdf``.
    :param out_dir: Directory the URDF and its ``meshes/`` tree are written to.
    :param hints: Resolutions recorded while a demo ran.
    :return: A report of what was written, shaped like
        :func:`bundle_urdf.bundle_urdf`'s.
    :raises FileNotFoundError: If the source itself cannot be found.
    """
    source_path = resolve_uri(source, hints=hints) or source
    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            "Gazebo source not found: %s (from %s)" % (source_path, source)
        )

    world = GazeboParser.from_file(source_path).parse()
    report = write_world_as_urdf(world, name, out_dir, GAZEBO_MESH_DIRECTORY)
    return {**report, "source": source_path}
