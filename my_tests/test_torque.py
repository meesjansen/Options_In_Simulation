from omni.isaac.kit import SimulationApp
import torch
import asyncio
import os
import time

# Configuration for livestream and headless mode
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": 3286,
}

# Start the omniverse application
simulation_app = SimulationApp(launch_config=CONFIG)

# Import necessary modules after simulation app initialization
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, print_stage_prim_paths
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils.extensions import enable_extension
import numpy as np


# Set livestream settings
simulation_app.set_setting("/app/window/drawMouse", True)
simulation_app.set_setting("/app/livestream/proto", "ws")
simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
simulation_app.set_setting("/ngx/enabled", False)

# Enable Native Livestream extension
enable_extension("omni.kit.livestream.native")

# Initialize the simulation world
world = World(stage_units_in_meters=1.0)

# Define the physics material for the ground plane
material = PhysicsMaterial(prim_path="/World/PhysicsMaterials", static_friction=0.5, dynamic_friction=0.5)
GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5])).apply_physics_material(material)

# Load the custom USD file for the robot
usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "origin_elevated.usd"))
prim_path = "/World/Robot"
add_reference_to_stage(usd_path, prim_path)

# List of wheel prim paths
wheel_prim_paths = [
    "/World/Robot/main_body/main_body_left_front_wheel",
    "/World/Robot/main_body/main_body_left_rear_wheel",
    "/World/Robot/main_body/main_body_right_front_wheel",
    "/World/Robot/main_body/main_body_right_rear_wheel",
]

# Print prim paths to debug
print_stage_prim_paths()

# Create an ArticulationView for the robot to retrieve DOF names
robot_articulations = ArticulationView(prim_paths_expr=prim_path, name="robot_view")
world.scene.add(robot_articulations)
world.reset()
robot_articulations.initialize()

# Retrieve and print the DOF names
dof_names = robot_articulations.dof_names
print("DOF Names from USD file:", dof_names)

# Wheel DOF names for applying torques
wheel_dof_names = [
    "left_front_wheel",
    "left_rear_wheel",
    "right_front_wheel",
    "right_rear_wheel"
]
print("Using predefined wheel DOF names:", wheel_dof_names)

wheel_torques = torch.tensor([100.0, 100.0, 100.0, 100.0])

while simulation_app._app.is_running() and not simulation_app.is_exiting():
    # Run in realtime mode, we don't specify the step size
    # Apply torques to the wheels
    robot_articulations.set_joint_efforts(wheel_torques, joint_indices=np.array([1,2,4,5]))
    simulation_app.update()

simulation_app.close()


