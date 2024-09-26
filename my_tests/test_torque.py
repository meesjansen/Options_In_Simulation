from omni.isaac.kit import SimulationApp

# Initialize the SimulationApp before any imports from omni.isaac.core
simulation_app = SimulationApp({"headless": True, "enable_livestream": True})


from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.physics_context import PhysicsContext

import torch
import numpy as np
import os
import asyncio
import time


# Initialize simulation world
world = World(stage_units_in_meters=1.0)

GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))

# Add the custom USD file to the stage
usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "limo_ackermann.usd"))  # Adjust this to the actual USD path for your Limo robot
prim_path = "/World/LimoRobot"

# Add the USD file reference to the simulation stage
add_reference_to_stage(usd_path, prim_path)

# Create an ArticulationView for the robot to retrieve DOF names
robot_articulations = ArticulationView(prim_paths_expr=prim_path, name="robot_view")

# Reset world to initialize everything
world.reset()

world.scene.add(robot_articulations)
robot_articulations.initialize()  # This line is critical

robot_articulations.set_solver_velocity_iteration_count(64)
robot_articulations.set_solver_position_iteration_count(32)

world.play()

# Retrieve and print the DOF names
dof_names = robot_articulations.dof_names
print("DOF Names from USD file:", dof_names)

wheel_dof_names = [
    "rear_left_wheel", 
    "rear_right_wheel",
    "front_left_wheel",
    "front_right_wheel"
]

# Load Limo robot instance
# robot = WheeledRobot(
#     prim_path=prim_path,  # You can adjust the prim path as needed
#     name="limo_robot",
#     wheel_dof_names=wheel_dof_names
# )

print("Using predefined wheel DOF names:", wheel_dof_names)

# Add robot to the world
# world.scene.add(robot)



# Define joint indices for the wheels (you can double-check them with the USD file)
# joint_indices = np.array([robot.get_dof_index(name) for name in wheel_dof_names], dtype=np.int32)  # Should return correct indices

def apply_wheel_torques(articulation_view, torques):
    articulation_view.set_joint_efforts(efforts=torques, joint_indices=[1, 2, 4, 5])

async def run_simulation():
    while simulation_app.is_running():
        await asyncio.sleep(0)  # Yield control to allow async event loop to continue

        # Step the simulation asynchronously
        world.step(render=False)

        # Apply torques to the wheels
        wheel_torques = torch.tensor([10.0, 10.0, 10.0, 10.0])
        apply_wheel_torques(robot_articulations, wheel_torques)

# Run the simulation asynchronously
asyncio.run(run_simulation())

# Close the simulation app when done
simulation_app.close()

# Close the simulation app when done
simulation_app.close()
