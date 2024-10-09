from omni.isaac.kit import SimulationApp

# Initialize the SimulationApp before any imports from omni.isaac.core
simulation_app = SimulationApp({"headless": True, "enable_livestream": True})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.physics_context import PhysicsContext
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.prims import GeometryPrim

import torch
import numpy as np
import os
import asyncio
import time


# Initialize simulation world
world = World(stage_units_in_meters=1.0)

material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials",
        static_friction=0.5,
        dynamic_friction=0.5,
    )

GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5])).apply_physics_material(material)

# Add the custom USD file to the stage
usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "origin_v10.usd"))  # Adjust this to the actual USD path for your Limo robot
prim_path = "/World/Robot"
add_reference_to_stage(usd_path, prim_path)

# Define the physics material properties for rubber
rubber_material = PhysicsMaterial(
    prim_path="/World/PhysicsMaterials/RubberMaterial",
    static_friction=0.9,    # Rubber has high static friction
    dynamic_friction=0.8,   # Slightly lower dynamic friction
    restitution=0.2         # Low restitution (rubber doesn't bounce much)
)

# Apply the physics material to each wheel
wheel_prim_paths = [
    "/World/Robot/left_front_wheel",
    "/World/Robot/left_rear_wheel",
    "/World/Robot/right_front_wheel",
    "/World/Robot/right_rear_wheel",
]

from omni.isaac.core.utils.stage import print_stage_prim_paths
print_stage_prim_paths()


# Add the USD file reference to the simulation stage
# add_reference_to_stage(usd_path, prim_path)

# Create an ArticulationView for the robot to retrieve DOF names
robot_articulations = ArticulationView(prim_paths_expr=prim_path, name="robot_view")

world.scene.add(robot_articulations)
# Reset world to initialize everything
world.reset()
robot_articulations.initialize()  # This line is critical



for wheel_path in wheel_prim_paths:
    # Define the geometry prim for each wheel
    wheel_prim = GeometryPrim(prim_path=wheel_path)
    # Apply the physics material to the wheel
    wheel_prim.apply_physics_material(rubber_material)


# Retrieve and print the DOF names
dof_names = robot_articulations.dof_names
print("DOF Names from USD file:", dof_names)

wheel_dof_names = [
    "left_front_wheel", 
    "left_rear_wheel",
    "right_front_wheel",
    "right_rear_wheel"
]

print("Using predefined wheel DOF names:", wheel_dof_names)


async def run_simulation():
    while simulation_app.is_running():
        await asyncio.sleep(0)  # Yield control to allow async event loop to continue

        # Apply torques to the wheels
        # wheel_torques = [100.0, 100.0, 100.0, 100.0]
        wheel_torques = torch.tensor([100.0, 100.0, 100.0, 100.0])
        robot_articulations.set_joint_efforts(wheel_torques) #, joint_indices=joint_indices)
        world.step(render=True)

    # steps = 100  # Run for 100 steps
    # for _ in range(steps):
    #     await asyncio.sleep(0)  # Yield control to allow async event loop to continue
    #     robot_articulations.set_joint_efforts(wheel_torques)
    #     world.step(render=True)




# Run the simulation asynchronously
asyncio.run(run_simulation())

# Close the simulation app when done
simulation_app.close()

# for i in range(500):
#     # Apply torques to the wheels
#     wheel_torques = torch.tensor([10.0, 10.0, 10.0, 10.0])
#     robot_articulations.set_joint_efforts(wheel_torques)
#     world.step(render=True)
# https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Documentation/Isaac-Sim-Docs_2021.2.1/app_isaacsim/app_isaacsim/tutorial_required_hello_world.html
# simulation_app.close()

