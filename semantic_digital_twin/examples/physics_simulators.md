---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
(physics-simulators)=
# Physics Simulators

This tutorial explains how to run physics simulations for a given world description.
We use **MuJoCo** as an example backend, but the same workflow applies to other physics engines supported by MultiSim.

# 1. Simulating a Predefined World

## 1.1 Required Imports

We begin by importing the necessary components:

* `MJCFParser` — parses a world description from an MJCF file.
* `MujocoSim` — runs the simulation.
* `SimulatorConstraints` — defines termination conditions.

```{code-cell} ipython3
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from base_simulator import SimulatorConstraints
import os
import time
```

## 1.2 Parsing a World Description

The world can either:

* Be loaded from a predefined MJCF file (recommended), or
* Be constructed manually (shown later in this tutorial).

Using predefined scenes is preferred because they are typically validated against the physics engine.

**Important**: Always validate your MJCF scene directly in MuJoCo before running it in MultiSim:

```bash
python -m mujoco.viewer --mjcf=path/to/your/scene.xml
```

Only a physically stable and functional scene can be expected to behave correctly inside MultiSim.

Below is a minimal example scene defined directly as an XML string.

```{code-cell} ipython3
if __name__ == "__main__":
    scene_xml_str = """
<mujoco>
    <worldbody>
        <body name="robot">
            <geom type="box" pos="0 0 0.5" size="0.2 0.2 0.5" rgba="0.9 0.9 0.9 1"/>
            <body name="left_shoulder" pos="0 0.3 0.9" quat="0.707 0.707 0 0">
                <joint name="left_shoulder_joint" type="hinge" axis="0 0 1"/>
                <geom type="cylinder" size="0.1 0.1 0.3" rgba="0.9 0.1 0.1 1"/>
                <body name="left_arm" pos="0 -0.4 -0.1" quat="0.707 0.707 0 0">
                    <joint name="left_arm_joint" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.3" rgba="0.1 0.9 0.1 1"/>
                </body>
            </body>
            <body name="right_shoulder" pos="0 -0.3 0.9" quat="0.707 0.707 0 0">
                <joint name="right_shoulder_joint" type="hinge" axis="0 0 1"/>
                <geom type="cylinder" size="0.1 0.1 0.3" rgba="0.9 0.1 0.1 1"/>
                <body name="right_arm" pos="0 -0.4 0.1" quat="0.707 0.707 0 0">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.3" rgba="0.1 0.9 0.1 1"/>
                </body>
            </body>
        </body>

        <body name="table" pos="0.5 0 0.25">
            <geom type="box" size="0.2 0.2 0.5" rgba="0.5 0.5 0.5 1"/>
        </body>

        <body name="object" pos="0.5 0 1.0">
            <freejoint/>
            <geom type="box" size="0.05 0.05 0.1" rgba="0.1 0.1 0.9 1"/>
        </body>
    </worldbody>
</mujoco>
"""
    world = MJCFParser.from_xml_string(scene_xml_str).parse()
```

This scene contains:

* A simple robot with two revolute arms
* A static table
* A dynamic object with a free joint

## 1.3 Running the Simulation

```{code-cell} ipython3
    headless = (
        os.environ.get("CI", "false").lower() == "true"
    )

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.001,
    )

    constraints = SimulatorConstraints(max_number_of_steps=10000)

    multi_sim.start_simulation(constraints=constraints)

    time_start = time.time()

    while multi_sim.is_running():
        time.sleep(0.1)
        print(
            f"Current number of steps: "
            f"{multi_sim.simulator.current_number_of_steps}"
        )

    print(f"Time elapsed: {time.time() - time_start:.2f}s")

    multi_sim.stop_simulation()
```

### Performance Considerations

This simple scene executes **10,000 steps in under 0.5 seconds**.

Simulation speed depends primarily on:

* Number of contacts
* Number of collision geometries
* Mesh complexity (vertex count)

For optimal performance:

* Prefer primitive shapes (boxes, cylinders, spheres)
* Avoid high-resolution meshes unless necessary

# 2. Building a World at Runtime

Instead of loading a static scene, the world can also be modified dynamically during simulation.

This is useful when:

* Constructing robots from semantic descriptions
* Spawning objects based on runtime conditions
* Generating environments procedurally

## 2.1 Required Imports

```{code-cell} ipython3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale, Color, Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
```

## 2.2 Spawning Bodies Programmatically

### Spawn Robot Base

```{code-cell} ipython3
def spawn_robot_body(spawn_world: World) -> Body:
    spawn_body = Body(name=PrefixedName("robot"))

    box_origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0, y=0, z=0.5,
        roll=0, pitch=0, yaw=0,
        reference_frame=spawn_body
    )

    box = Box(
        origin=box_origin,
        scale=Scale(0.4, 0.4, 1.0),
        color=Color(0.9, 0.9, 0.9, 1.0),
    )

    spawn_body.collision = ShapeCollection(
        [box], reference_frame=spawn_body
    )

    with spawn_world.modify_world():
        spawn_world.add_connection(
            FixedConnection(
                parent=spawn_world.root,
                child=spawn_body
            )
        )

    return spawn_body
```

### Spawn Shoulder Joints

```{code-cell} ipython3
def spawn_shoulder_bodies(
    spawn_world: World,
    root_body: Body
) -> tuple[Body, Body]:

    # Left shoulder
    spawn_left_shoulder_body = Body(
        name=PrefixedName("left_shoulder")
    )

    cylinder = Cylinder(
        width=0.2,
        height=0.1,
        color=Color(0.9, 0.1, 0.1, 1.0),
    )

    spawn_left_shoulder_body.collision = ShapeCollection(
        [cylinder],
        reference_frame=spawn_left_shoulder_body
    )

    dof = DegreeOfFreedom(
        name=PrefixedName("left_shoulder_joint")
    )

    left_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0, pos_y=0.3, pos_z=0.9,
        quat_w=0.707, quat_x=0.707,
        quat_y=0, quat_z=0
    )

    with spawn_world.modify_world():
        spawn_world.add_degree_of_freedom(dof)
        spawn_world.add_connection(
            RevoluteConnection(
                name=dof.name,
                parent=root_body,
                child=spawn_left_shoulder_body,
                axis=Vector3.Z(reference_frame=spawn_left_shoulder_body),
                dof_id=dof.id,
                parent_T_connection_expression=left_origin,
            )
        )

    # Right shoulder
    spawn_right_shoulder_body = Body(
        name=PrefixedName("right_shoulder")
    )

    spawn_right_shoulder_body.collision = ShapeCollection(
        [cylinder],
        reference_frame=spawn_right_shoulder_body
    )

    dof = DegreeOfFreedom(
        name=PrefixedName("right_shoulder_joint")
    )

    right_origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0, pos_y=-0.3, pos_z=0.9,
        quat_w=0.707, quat_x=0.707,
        quat_y=0, quat_z=0
    )

    with spawn_world.modify_world():
        spawn_world.add_degree_of_freedom(dof)
        spawn_world.add_connection(
            RevoluteConnection(
                name=dof.name,
                parent=root_body,
                child=spawn_right_shoulder_body,
                axis=Vector3.Z(reference_frame=spawn_right_shoulder_body),
                dof_id=dof.id,
                parent_T_connection_expression=right_origin,
            )
        )

    return spawn_left_shoulder_body, spawn_right_shoulder_body
```

## 2.3 Spawning During Simulation

```{code-cell} ipython3
if __name__ == "__main__":
    world = World()

    headless = (
        os.environ.get("CI", "false").lower() == "true"
    )

    multi_sim = MujocoSim(
        world=world,
        headless=headless,
        step_size=0.001,
    )

    constraints = SimulatorConstraints(
        max_number_of_steps=1_000_000
    )

    multi_sim.start_simulation(constraints=constraints)

    time_start = time.time()

    while multi_sim.is_running():
        if multi_sim.simulator.current_number_of_steps == 100:

            time_spawn_start = time.time()

            robot_body = spawn_robot_body(world)

            spawn_shoulder_bodies(
                spawn_world=world,
                root_body=robot_body
            )

            print(
                f"Time to spawn bodies: "
                f"{time.time() - time_spawn_start:.2f}s"
            )

    print(f"Time elapsed: {time.time() - time_start:.2f}s")

    multi_sim.stop_simulation()
```

### Key Property

Objects are spawned **without resetting the simulation**.
The physics state remains continuous, and the world is modified dynamically at runtime.

This allows:

* Incremental scene construction
* Online model updates
* Dynamic world adaptation