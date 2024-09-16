import os
import torch
import numpy as np

from typing import Optional

from omni.isaac.kit import SimulationApp
kit = SimulationApp({"headless": True})

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.prims import get_prim_at_path

from pxr import PhysxSchema

class LimoAckermann(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "limo_ackermann",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """Initialize the LimoAckermann robot with the appropriate drives and DoFs."""
        
        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        if self._usd_path is None:
            self._usd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "limo_ackermann.usd"))

        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        # Define the wheels and their corresponding DOF paths
        self.wheel_dof_paths = [
            "left_steering_hinge/front_left_wheel",
            "right_steering_hinge/front_right_wheel",
            "rear_left_wheel_link/rear_left_wheel",
            "rear_right_wheel_link/rear_right_wheel"
        ]

        # Initialize dynamic control API
        self.dc = _dynamic_control.acquire_dynamic_control_interface()

        # Retrieve the articulation for the robot
        self.art = self.dc.get_articulation(self.prim_path)
        self.dc.wake_up_articulation(self.art)

        # Number of degrees of freedom (joints) of the robot
        self.num_dofs = self.dc.get_articulation_dof_count(self.art)

        # Initialize joint efforts to zero torque
        self.joint_efforts = torch.zeros(self.num_dofs, device=self._device)

    def apply_torque(self, torque_values):
        """Apply the torque values to the robot's joints.

        Args:
            torque_values (torch.Tensor): A tensor containing torque values for each joint.
        """
        self.dc.apply_articulation_dof_efforts(self.art, torque_values.cpu().numpy())

    
    # Since the RL primitive actions are not absolute torque values but relative changes the 
    # application of actions should be defined, the same goes for the macro-actions
    
    
    
    
    # def apply_action(self, action):
    #     """Apply a primitive action to the robot's wheels.
        
    #     Args:
    #         action (torch.Tensor): A tensor of shape (4,) where each element is -1 (decrease), 0 (maintain), or 1 (increase).
    #     """
    #     torque_step = 5.0  # Nm, example step

    #     for i, dof in enumerate(self.wheel_dof_paths):
    #         current_torque = self.articulation_controller.get_dof_state(dof)["effort"]
    #         new_torque = current_torque + action[i].item() * torque_step
    #         self.articulation_controller.set_dof_effort(dof, new_torque)

    # def apply_macro_action(self, macro_action, steps):
    #     """Apply a macro action to the robot's wheels over a number of steps.
        
    #     Args:
    #         macro_action (str): Type of macro action ("straight", "turn_left", "turn_right").
    #         steps (int): Number of physics steps to apply this macro action.
    #     """
    #     torque_step = 5.0  # Nm, example step
    #     current_torques = []

    #     # Retrieve the current torque for each wheel
    #     for i, dof in enumerate(self.wheel_dof_paths):
    #         current_torque = self.articulation_controller.get_dof_state(dof)["effort"]
    #         current_torques.append(current_torque)

    #     if macro_action == "straight":
    #         # Average the current torques to maintain straight movement
    #         average_torque = sum(current_torques) / len(current_torques)
    #         action = torch.tensor([average_torque, average_torque, average_torque, average_torque])

    #         for _ in range(steps):
    #             for i, dof in enumerate(self.wheel_dof_paths):
    #                 self.articulation_controller.set_dof_effort(dof, action[i])
    #             self.step_simulation()

    #     elif macro_action == "turn_left":
    #         # Differential torque for turning left
    #         average_torque = sum(current_torques) / len(current_torques)
    #         action = torch.tensor([
    #             average_torque + torque_step,  # Increase torque on right wheels
    #             average_torque + torque_step,  
    #             average_torque - torque_step,  # Decrease torque on left wheels
    #             average_torque - torque_step  
    #         ])

    #         for _ in range(steps):
    #             for i, dof in enumerate(self.wheel_dof_paths):
    #                 self.articulation_controller.set_dof_effort(dof, action[i])
    #             self.step_simulation()

    #     elif macro_action == "turn_right":
    #         # Differential torque for turning right
    #         average_torque = sum(current_torques) / len(current_torques)
    #         action = torch.tensor([
    #             average_torque - torque_step,  # Decrease torque on right wheels
    #             average_torque - torque_step,  
    #             average_torque + torque_step,  # Increase torque on left wheels
    #             average_torque + torque_step  
    #         ])

    #         for _ in range(steps):
    #             for i, dof in enumerate(self.wheel_dof_paths):
    #                 self.articulation_controller.set_dof_effort(dof, action[i])
    #             self.step_simulation()

    # def step_simulation(self):
    #     """Advance the simulation by one timestep.
    #     """
    #     # Here you would include the logic to advance the simulation
    #     # This could involve interacting with the environment's step function
    #     pass