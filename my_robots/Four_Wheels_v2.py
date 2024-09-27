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



class LimoAckermann(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "limo",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the Limo robot with the appropriate DoFs."""
        
        self._usd_path = usd_path
        self._name = name
        # self.device = self.device

        if self._usd_path is None:
            self._usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "limo_ackermann.usd"))

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
                "rear_left_wheel", 
                "rear_right_wheel",
                "front_left_wheel",
                "front_right_wheel"
        ]

        
        
        
        # Convert the list of indices to a PyTorch tensor
        # self._dof_indices = torch.tensor([self.get_dof_index(dof) for dof in self._dof_names], dtype=torch.int64, device=self.device)



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