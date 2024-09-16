import torch

class FourWheeledPlatform:
    def __init__(self, torque_value=1.0, turn_differential=0.5, device="cpu"):
        """
        Initialize the four-wheeled platform.

        :param torque_value: Base torque value for straight driving.
        :param turn_differential: Differential torque value for turning.
        :param device: The device to store tensors ("cpu" or "cuda").
        """
        self.torque_value = torque_value
        self.turn_differential = turn_differential
        self.device = device

        # Define the macro-actions as dictionaries with action names as keys and corresponding torque values as values
        self.macro_actions = {
            "straight_drive": self.straight_drive,
            "turn": self.turn
        }

    def straight_drive(self):
        """
        Macro-action for driving straight with a constant torque.
        
        :return: A tensor representing torques applied to the four wheels [front_left, front_right, rear_left, rear_right]
        """
        # Apply the same torque to all wheels for straight driving
        action = torch.tensor([self.torque_value, self.torque_value, self.torque_value, self.torque_value], device=self.device)
        return action

    def turn(self, direction="left"):
        """
        Macro-action for turning the platform. The turn is achieved by applying a differential torque.

        :param direction: Direction of the turn ("left" or "right").
        :return: A tensor representing torques applied to the four wheels [front_left, front_right, rear_left, rear_right]
        """
        if direction == "left":
            # Reduce torque on the left wheels and increase torque on the right wheels
            action = torch.tensor(
                [self.torque_value - self.turn_differential, self.torque_value + self.turn_differential,
                 self.torque_value - self.turn_differential, self.torque_value + self.turn_differential], 
                device=self.device)
        elif direction == "right":
            # Reduce torque on the right wheels and increase torque on the left wheels
            action = torch.tensor(
                [self.torque_value + self.turn_differential, self.torque_value - self.turn_differential,
                 self.torque_value + self.turn_differential, self.torque_value - self.turn_differential], 
                device=self.device)
        else:
            raise ValueError("Invalid direction for turn. Must be 'left' or 'right'.")

        return action

    def get_macro_action(self, action_name, *args):
        """
        Retrieve the macro-action by its name and return the corresponding action tensor.
        
        :param action_name: The name of the macro-action ("straight_drive" or "turn").
        :param args: Additional arguments required by the macro-action (e.g., direction for turning).
        :return: A tensor representing the action to be applied to the four wheels.
        """
        if action_name not in self.macro_actions:
            raise ValueError(f"Macro-action '{action_name}' is not defined.")

        return self.macro_actions[action_name](*args)

