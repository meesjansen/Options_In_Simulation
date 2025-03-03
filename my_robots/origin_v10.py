import os
import torch
import numpy as np
from typing import Optional
from pxr import PhysxSchema

from omni.isaac.kit import SimulationApp
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.materials.physics_material import PhysicsMaterial

class AvularOrigin_v10(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "origin_v10",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the robot and enable custom collision geometry for its wheels."""
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "my_assets/urdf/origin_v10", "origin_v10.usd")
            )

        # Add the USD reference to the stage
        add_reference_to_stage(self._usd_path, prim_path)

        # Initialize the robot using the parent class
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        # Define the wheels (DOF names) for your robot
        self._dof_names = [
            "left_front_wheel",
            "left_rear_wheel",
            "right_front_wheel",
            "right_rear_wheel",
        ]

        # Define the collision prim paths for each wheel.
        # Adjust the paths to match your USD hierarchy (here we assume your robot is loaded at prim_path).
        wheel_collision_paths = [
            f"{prim_path}/left_front_wheel/left_front_wheel_collision",
            f"{prim_path}/right_front_wheel/right_front_wheel_collision",
            f"{prim_path}/left_rear_wheel/left_rear_wheel_collision",
            f"{prim_path}/right_rear_wheel/right_rear_wheel_collision"
        ]

        # Enable the custom collision geometry for each wheel
        for path in wheel_collision_paths:
            AvularOrigin_v10.enable_custom_geometry_for_collision(path)

    @staticmethod
    def enable_custom_geometry_for_collision(collision_prim_path: str):
        """
        Retrieves the collision prim at the given path and enables the custom geometry flag,
        so that Isaac Sim uses the native cylinder shape for collisions.
        """
        collision_prim = get_prim_at_path(collision_prim_path)
        if collision_prim is not None and collision_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Get(collision_prim)
            physx_collision_api.GetCustomGeometryAttr().Set(True)
            print(f"Custom geometry enabled for: {collision_prim_path}")
        else:
            print(f"Prim at {collision_prim_path} does not have PhysxCollisionAPI.")

    @property
    def dof_names(self):
        return self._dof_names

    def set_robot_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64 / np.pi * 180)
