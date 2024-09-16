from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gym
import gymnasium

import torch

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

# Configuration dictionary
Q_LEARNING_DEFAULT_CONFIG = {
    "discount_factor": 0.99,        # discount factor (gamma)

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "learning_rate": 0.5,           # learning rate (alpha)

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "macro_action_duration": 5,     # duration of each macro-action in timesteps

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}

class MacroQ_LEARNING(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Q-learning with Macro-Actions and Primitive Actions

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(Q_LEARNING_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        # configuration
        self._discount_factor = self.cfg["discount_factor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._learning_rate = self.cfg["learning_rate"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._macro_action_duration = self.cfg["macro_action_duration"]

        # create temporary variables needed for storage and computation
        self._current_states = None
        self._current_actions = None
        self._current_rewards = None
        self._current_next_states = None
        self._current_dones = None

        # Macro-action tracking
        self._current_macro_action = None
        self._macro_action_remaining = 0
        self._macro_action_accumulated_reward = 0

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)

    def _is_macro_action(self, action: torch.Tensor) -> bool:
        """Determine if the chosen action is a macro-action"""
        # For now, we assume the action tensor indicates whether it's a macro-action
        # This can be customized as needed, once number of prims and macros is defined
        return action.item() >= 0  # Example condition, adjust as needed

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        if self._macro_action_remaining > 0:
            # Continue with the current macro-action
            self._macro_action_remaining -= 1
            return self._current_macro_action

        # Choose a new action (either macro-action or primitive)
        new_action = self.policy.act({"states": states}, role="policy")
        if self._is_macro_action(new_action):
            self._current_macro_action = new_action
            self._macro_action_remaining = self._macro_action_duration - 1
            self._macro_action_accumulated_reward = 0  # Reset accumulated reward
        else:
            # Handle primitive action (single timestep)
            self._current_macro_action = new_action
            self._macro_action_remaining = 0  # Primitive actions have no duration

        return self._current_macro_action

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        # Accumulate rewards for the macro-action
        self._macro_action_accumulated_reward += rewards.sum().item()

        # If the macro-action is completed, update the Q-table
        if self._macro_action_remaining == 0:
            if self._rewards_shaper is not None:
                self._macro_action_accumulated_reward = self._rewards_shaper(self._macro_action_accumulated_reward, timestep, timesteps)

            self._current_states = states
            self._current_actions = actions
            self._current_rewards = torch.tensor([self._macro_action_accumulated_reward], device=self.device)
            self._current_next_states = next_states
            self._current_dones = terminated + truncated

            if self.memory is not None:
                self.memory.add_samples(states=states, actions=actions, rewards=self._current_rewards, next_states=next_states,
                                        terminated=terminated, truncated=truncated)
                for memory in self.secondary_memories:
                    memory.add_samples(states=states, actions=actions, rewards=self._current_rewards, next_states=next_states,
                                       terminated=terminated, truncated=truncated)

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts and self._macro_action_remaining == 0:
            self._update(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # Get the Q-table from the policy model
        q_table = self.policy.table()
        env_ids = torch.arange(self._current_rewards.shape[0]).view(-1, 1)

        # Compute the next actions by selecting the one with the highest Q-value
        next_actions = torch.argmax(q_table[env_ids, self._current_next_states], dim=-1, keepdim=True).view(-1, 1)

        # Update Q-table using the correct update rule
        q_table[env_ids, self._current_states, self._current_actions] += self._learning_rate \
            * (self._current_rewards + self._discount_factor * self._current_dones.logical_not() \
                * q_table[env_ids, self._current_next_states, next_actions] \
                    - q_table[env_ids, self._current_states, self._current_actions])
