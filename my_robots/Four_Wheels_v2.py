import os
import torch
import numpy as np

from typing import Optional

from omni.isaac.kit import SimulationApp
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.wheeled_robots.robots import WheeledRobot


from pxr import PhysxSchema

class LimoAckermann(WheeledRobot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "limo_ackermann",
        usd_path: Optional[str] = None,
        wheel_dof_names: Optional[str] = None,
        position: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """Initialize the LimoAckermann robot with the appropriate drives and DoFs."""
        
        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.0, 0.0, 0.0]) if position is None else position
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        if self._usd_path is None:
            self._usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_assets", "limo_ackermann.usd"))

        add_reference_to_stage(self._usd_path, prim_path)

        # Define the wheels and their corresponding DOF paths
        self.wheel_dof_paths = [
            "front_left_wheel_link/front_left_wheel",
            "front_right_wheel_link/front_right_wheel",
            "rear_left_wheel_link/rear_left_wheel",
            "rear_right_wheel_link/rear_right_wheel"
        ]

        self._dof_names = [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel"
        ]
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=self._dof_names,
            position=self._position,
            orientation=self._orientation
        )
        

    @property
    def dof_names(self):
        return self._dof_names
        
    def apply_torque(self, torque_values):
        """Apply the torque values to the robot's joints"""

        self.set_joint_efforts(torque_values) 