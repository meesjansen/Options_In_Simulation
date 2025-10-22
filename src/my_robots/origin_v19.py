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

import my_assets
from my_assets import origin_v19_usd  # or `usd_path` if you prefer the generic one

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
            with origin_v19_usd() as p:
                self._usd_path = str(p)

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
                "front_left_wheel",
                "front_right_wheel",
                "rear_left_wheel",
                "rear_right_wheel",
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

            # collisions_prim = link_prim.GetChild("collisions")
            # if collisions_prim and collisions_prim.IsValid():
            #     # Ensure PhysxCollisionAPI is applied
            #     if not PhysxSchema.PhysxCollisionAPI.Has(collisions_prim):
            #         PhysxSchema.PhysxCollisionAPI.Apply(collisions_prim)

            #     collision_api = PhysxSchema.PhysxCollisionAPI.Get(collisions_prim)
            #     collision_api.GetCustomGeometryAttr().Set(True)
            #     print(f"Custom geometry enabled for: {collisions_prim.GetPath()}")