from typing import Optional

import os
import torch
import numpy as np

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

class Iiwa14(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "iiwa14",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        if self._usd_path is None:
            self._usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "iiwa14.usd"))

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "world/iiwa_joint_1",
            "iiwa_link_1/iiwa_joint_2",
            "iiwa_link_2/iiwa_joint_3",
            "iiwa_link_3/iiwa_joint_4",
            "iiwa_link_4/iiwa_joint_5",
            "iiwa_link_5/iiwa_joint_6",
            "iiwa_link_6/iiwa_joint_7",
        ]

        drive_type = ["angular"] * 7
        default_dof_pos = [0, 0, 0, -90, 0, 90, 0]  # degrees
        stiffness = [500.0 * np.pi / 180] * 7
        damping = [100.0 * np.pi / 180] * 7
        max_force = [320, 320, 176, 176, 110, 40, 40]  # Nm
        max_velocity = [85, 85, 100, 75, 130, 135, 135]  # degrees/s

        for i, dof in enumerate(dof_paths):
            print(f"{self.prim_path}/{dof}")
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])
