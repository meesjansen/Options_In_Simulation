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
from omni.isaac.wheeled_robots.robots import WheeledRobot
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
        """Initialize the Limo robot with the appropriate DoFs."""
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "origin_v10.usd"))

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        # Define the wheels and their corresponding DOF paths
        self._dof_names = [
                "left_front_wheel",
                "left_rear_wheel",
                "right_front_wheel",
                "right_rear_wheel",
        ]


        
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

