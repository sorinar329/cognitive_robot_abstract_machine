"""
Make the real HSRB URDF (``hsr_description``, the ROS package
:meth:`~semantic_digital_twin.robots.hsrb.HSRB.get_ros_file_path` points at) resolvable
and parseable in an environment that has never built a ROS workspace containing it, so
the standalone HSRB demos in this package can spawn the real robot description without
requiring a full ``colcon``/``catkin`` build.

Three separate problems block that on a bare checkout:

1. :class:`~semantic_digital_twin.adapters.package_resolver.CompositePathResolver`
   resolves ``package://`` URIs via ``ament_index_python``, which only finds a package
   once it is registered in some ``AMENT_PREFIX_PATH`` entry's resource index -- a
   ``colcon build`` side effect, not something a bare source checkout has. The xacro
   processor's own internal ``$(find hsr_description)``/``$(find hsr_meshes)`` calls
   (used for its ``<xacro:include>`` and mesh paths) go through that same ament lookup,
   so even pointing :meth:`~semantic_digital_twin.robots.hsrb.HSRB.get_ros_file_path`'s
   top-level file directly at the checkout is not enough on its own.
2. ``hsr_description``'s own xacro sources declare three dummy, zero-range mounting
   frames (front/back bumper mounts, the wrist force-torque sensor frame) as
   ``type="revolute"`` joints with no ``<axis>`` tag at all -- both a URDF parser
   compatibility problem (this repo's, :mod:`semantic_digital_twin.adapters.urdf`,
   asserts an axis must be present, unlike some URDF parsers, which fall back to the
   spec's implied ``(1, 0, 0)`` default) and, once given a synthetic axis to parse at
   all, a MuJoCo stability problem: each mounted link has negligible mass/geometry of
   its own, and a real, actuatable joint connecting to a near-massless body is
   numerically degenerate (observed to drive ``QACC`` to ``NaN`` within the first
   physics step, however that joint is otherwise held or left alone). Retyped to
   ``type="fixed"`` instead -- what a zero-range dummy mounting frame actually is --
   sidesteps both: a fixed joint needs no axis, and MuJoCo welds it rigidly to its
   parent rather than giving it independent dynamics.
3. ``arm_flex_link``'s declared inertia tensor has ``ixx="7.528"`` where its neighbours
   ``iyy``/``izz`` are two orders of magnitude smaller (``0.007102``/``0.001552``) -- an
   evident decimal-point typo (the physically implausible value describes a link with
   the rotational inertia of a small building) that also happens to violate the
   triangle inequality every physically realizable inertia tensor must satisfy, which
   MuJoCo's own ``mjSpec.compile()`` enforces and rejects.

:func:`ensure_hsrb_description_available` fixes all three, without touching the user's
own ``hsr_description``/``hsr_meshes`` checkout (its on-disk state is left alone; the
patched copy lives in a separate cache directory): it copies the two packages into a
persistent cache, applies :data:`SOURCE_PATCHES`, builds a minimal ``ament_index``
resource-index entry for both, and adds that cache to ``AMENT_PREFIX_PATH`` for the
current process.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from typing_extensions import Iterable, Tuple

CACHE_DIRECTORY = Path.home() / ".cache" / "cognitive_robot_abstract_machine"
"""
Where the patched copy of ``hsr_description``/``hsr_meshes`` and its ament resource
index live, persisted across runs so :func:`ensure_hsrb_description_available` only
copies and patches once.
"""

PATCHED_HSR_DESCRIPTION_DIRECTORY = CACHE_DIRECTORY / "hsr_description_patched"
"""
Root of the patched copy: ``share/hsr_description``, ``share/hsr_meshes``, and the
``share/ament_index/resource_index/packages`` entries registering both.
"""

SOURCE_PACKAGE_NAMES = ("hsr_description", "hsr_meshes")
"""
The two ROS packages copied out of :data:`SOURCE_RESOURCES_DIRECTORY` into
:data:`PATCHED_HSR_DESCRIPTION_DIRECTORY`.
"""

SOURCE_RESOURCES_DIRECTORY = Path.home() / "sim_ws" / "resources"
"""
Where this machine's real, unpatched ``hsr_description``/``hsr_meshes`` checkouts live,
read but never written to by this module.
"""

SOURCE_PATCHES: Tuple[Tuple[str, str, str], ...] = (
    (
        "hsr_description/urdf/base_v2/base.urdf.xacro",
        '<joint name="${prefix}_f_bumper_joint" type="revolute">\n'
        '            <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/>\n'
        '            <parent link="${prefix}_link"/>\n'
        '            <child link="${prefix}_f_bumper_link"/>\n'
        '            <limit effort="0" lower="0" upper="0" velocity="0"/>\n'
        "        </joint>",
        '<joint name="${prefix}_f_bumper_joint" type="fixed">\n'
        '            <origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/>\n'
        '            <parent link="${prefix}_link"/>\n'
        '            <child link="${prefix}_f_bumper_link"/>\n'
        "        </joint>",
    ),
    (
        "hsr_description/urdf/base_v2/base.urdf.xacro",
        '<joint name="${prefix}_b_bumper_joint" type="revolute">\n'
        '            <origin rpy="0.0 0.0 3.141592653589793" xyz="0.0 0.0013 0.0"/>\n'
        '            <parent link="${prefix}_link"/>\n'
        '            <child link="${prefix}_b_bumper_link"/>\n'
        '            <limit effort="0" lower="0" upper="0" velocity="0"/>\n'
        "        </joint>",
        '<joint name="${prefix}_b_bumper_joint" type="fixed">\n'
        '            <origin rpy="0.0 0.0 3.141592653589793" xyz="0.0 0.0013 0.0"/>\n'
        '            <parent link="${prefix}_link"/>\n'
        '            <child link="${prefix}_b_bumper_link"/>\n'
        "        </joint>",
    ),
    (
        "hsr_description/urdf/sensors/ft_sensor.urdf.xacro",
        '<joint name="${prefix}_ft_sensor_frame_joint" type="revolute">\n'
        '            <xacro:insert_block name="origin"/>\n'
        '            <parent link="${parent}"/>\n'
        '            <child link="${prefix}_ft_sensor_frame"/>\n'
        '            <limit effort="100.0" lower="0" upper="0" velocity="1.5"/>\n'
        "        </joint>",
        '<joint name="${prefix}_ft_sensor_frame_joint" type="fixed">\n'
        '            <xacro:insert_block name="origin"/>\n'
        '            <parent link="${parent}"/>\n'
        '            <child link="${prefix}_ft_sensor_frame"/>\n'
        "        </joint>",
    ),
    (
        "hsr_description/urdf/arm_v0/arm.urdf.xacro",
        '<inertia ixx="7.528" ixy="-0.000020284207" ixz="-0.000022947194"\n'
        '                    iyy="0.007102" iyz="-0.000091796075" izz="0.001552"/>',
        '<inertia ixx="0.007528" ixy="-0.000020284207" ixz="-0.000022947194"\n'
        '                    iyy="0.007102" iyz="-0.000091796075" izz="0.001552"/>',
    ),
)
"""
``(relative_file, old_text, new_text)`` triples fixing the three joints described in
this module's docstring that are declared ``type="revolute"`` but zero-range and
missing an ``<axis>`` this repo's URDF parser requires -- retyped to ``fixed`` instead
of merely adding the missing axis, since a zero-range dummy mounting frame is exactly
what ``fixed`` describes, and MuJoCo welds a fixed-jointed body rigidly to its parent
rather than giving it its own (here, near-massless and numerically degenerate)
dynamics -- plus the ``arm_flex_link`` inertia typo, also described there. Each
``old_text`` includes enough surrounding context to identify the right, unique
replacement point.
"""


def _copy_source_packages() -> None:
    """
    Copy :data:`SOURCE_PACKAGE_NAMES` from :data:`SOURCE_RESOURCES_DIRECTORY` into
    :data:`PATCHED_HSR_DESCRIPTION_DIRECTORY`'s ``share`` directory, replacing whatever
    copy is already cached there.
    """
    share_directory = PATCHED_HSR_DESCRIPTION_DIRECTORY / "share"
    share_directory.mkdir(parents=True, exist_ok=True)
    for package_name in SOURCE_PACKAGE_NAMES:
        source = SOURCE_RESOURCES_DIRECTORY / package_name
        if not source.is_dir():
            raise FileNotFoundError(
                f"Expected the real '{package_name}' checkout at '{source}'; it does "
                "not exist. Point SOURCE_RESOURCES_DIRECTORY at wherever this "
                "machine's hsr_description/hsr_meshes actually live."
            )
        destination = share_directory / package_name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def _apply_source_patches() -> None:
    """
    Apply every :data:`SOURCE_PATCHES` triple to the cached copy, raising if a patch's
    ``old_text`` is not found (e.g. because the upstream source changed and this
    module's patches need updating too).
    """
    share_directory = PATCHED_HSR_DESCRIPTION_DIRECTORY / "share"
    for relative_file, old_text, new_text in SOURCE_PATCHES:
        target = share_directory / relative_file
        content = target.read_text()
        if old_text not in content:
            raise ValueError(
                f"Expected text not found in '{target}'; hsr_description's source may "
                "have changed since SOURCE_PATCHES was written."
            )
        target.write_text(content.replace(old_text, new_text, 1))


def _build_ament_resource_index() -> None:
    """
    Register :data:`SOURCE_PACKAGE_NAMES` as discoverable ``ament_index`` packages
    rooted at :data:`PATCHED_HSR_DESCRIPTION_DIRECTORY`, so both
    :class:`~semantic_digital_twin.adapters.package_resolver.AmentPackageLocator` and
    xacro's own internal ``$(find ...)`` calls resolve them without a real
    ``colcon build``.
    """
    share_directory = PATCHED_HSR_DESCRIPTION_DIRECTORY / "share"
    resource_index_directory = (
        share_directory / "ament_index" / "resource_index" / "packages"
    )
    resource_index_directory.mkdir(parents=True, exist_ok=True)
    for package_name in SOURCE_PACKAGE_NAMES:
        (resource_index_directory / package_name).touch()


def _cache_is_provisioned() -> bool:
    """
    Whether :data:`PATCHED_HSR_DESCRIPTION_DIRECTORY` already holds a copy with every
    :data:`SOURCE_PATCHES` patch applied, so :func:`ensure_hsrb_description_available`
    can skip re-copying and re-patching it.
    """
    share_directory = PATCHED_HSR_DESCRIPTION_DIRECTORY / "share"
    for package_name in SOURCE_PACKAGE_NAMES:
        if not (share_directory / package_name).is_dir():
            return False
    for relative_file, _, new_text in SOURCE_PATCHES:
        target = share_directory / relative_file
        if not target.is_file() or new_text not in target.read_text():
            return False
    return True


def _prepend_to_ament_prefix_path(paths: Iterable[Path]) -> None:
    """
    Prepend ``paths`` to the current process's ``AMENT_PREFIX_PATH``, without
    duplicating an entry already present.

    :param paths: Directories to add, in order.
    """
    existing = os.environ.get("AMENT_PREFIX_PATH", "")
    existing_entries = [entry for entry in existing.split(os.pathsep) if entry]
    new_entries = [str(path) for path in paths if str(path) not in existing_entries]
    os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(new_entries + existing_entries)


def ensure_hsrb_description_available() -> None:
    """
    Make ``package://hsr_description/...`` and ``package://hsr_meshes/...`` resolvable
    and parseable for the rest of this process, provisioning
    :data:`PATCHED_HSR_DESCRIPTION_DIRECTORY` first if it is not already cached from a
    previous run.

    Call this before parsing the real HSRB URDF (e.g. before
    ``URDFParser.from_file(HSRB.get_ros_file_path())``), and before importing anything
    that resolves it as a side effect.
    """
    if not _cache_is_provisioned():
        _copy_source_packages()
        _apply_source_patches()
        _build_ament_resource_index()
    _prepend_to_ament_prefix_path([PATCHED_HSR_DESCRIPTION_DIRECTORY])
