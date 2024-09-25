from omni.isaac.kit import SimulationApp

# Initialize the SimulationApp before any imports from omni.isaac.core
simulation_app = SimulationApp({"headless": True, "enable_livestream": True, "enable_viewport": True})


from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.wheeled_robots.robots import WheeledRobot

import torch
import numpy as np
import os


# Initialize simulation world
world = World(stage_units_in_meters=1.0)

# Add the custom USD file to the stage
usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "limo_ackermann.usd"))  # Adjust this to the actual USD path for your Limo robot
prim_path = "/World/LimoRobot"

# Add the USD file reference to the simulation stage
add_reference_to_stage(usd_path, prim_path)

# Create an ArticulationView for the robot to retrieve DOF names
articulation_view = ArticulationView(prim_paths_expr=prim_path)

# Reset world to initialize everything
world.reset()

# Step once to ensure physics and articulation are loaded
world.step(render=False)

# Initialize the ArticulationView
articulation_view.initialize()  # This line is critical

# Retrieve and print the DOF names
dof_names = articulation_view.dof_names()
print("DOF Names from USD file:", dof_names)

wheel_dof_names = [
    "front_left_wheel_link/front_left_wheel", 
    "front_right_wheel_link/front_right_wheel",
    "rear_left_wheel_link/rear_left_wheel", 
    "rear_right_wheel_link/rear_right_wheel"
]

# Load Limo robot instance
robot = WheeledRobot(
    prim_path=prim_path,  # You can adjust the prim path as needed
    name="limo_robot",
    wheel_dof_names=wheel_dof_names
)

print("Using predefined wheel DOF names:", wheel_dof_names)

# Add robot to the world
world.scene.add(robot)

# Reset world to initialize everything
world.reset()

# Define joint indices for the wheels (you can double-check them with the USD file)
joint_indices = [robot.wheel_dof_indices[name] for name in robot.wheel_dof_names]  # Should return correct indices

# Function to set torques to the robot's wheels
def apply_wheel_torques(robot, torques):
    # Set torques to the robot's wheels
    robot.apply_torque(joint_indices, torques)

# Run the simulation loop
while simulation_app.is_running():
    # Step the simulation
    world.step(render=True)

    # Apply torques to the wheels (for this example, let's drive the robot forward)
    # You can try different values to see the effect
    wheel_torques = torch.tensor([10.0, 10.0, 10.0, 10.0])  # Simple forward driving torques for all wheels
    apply_wheel_torques(robot, wheel_torques)

    # Break the loop if the simulation is done
    if not simulation_app.is_running():
        break

# Close the simulation app when done
simulation_app.close()
