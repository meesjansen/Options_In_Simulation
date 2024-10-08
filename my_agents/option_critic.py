from typing import Any, Mapping, Optional, Tuple, Union

import copy
import math
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

class OptionCriticAgent(Agent):
    """Option-Critic Agent for the skrl library

    This implementation simplifies the architecture by using a single network
    to output the policy over options, intra-option policies, termination probabilities,
    and option values. It also streamlines the update mechanism for better efficiency.
    """

    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Initialize the agent

        :param models: Models used by the agent. Required key: 'model'
        :type models: dict of str: skrl.models.torch.Model
        :param memory: Memory object for experience replay
        :type memory: skrl.memories.torch.Memory
        :param cfg: Configuration dictionary with hyperparameters
        :type cfg: dict
        """
        super().__init__(models=models, memory=memory, cfg=cfg)

        # Configuration and hyperparameters
        self.learning_rate = self.cfg.get("learning_rate", 1e-3)
        self.discount_factor = self.cfg.get("discount_factor", 0.99)
        self.entropy_coefficient = self.cfg.get("entropy_coefficient", 0.01)
        self.batch_size = self.cfg.get("batch_size", 64)
        self.num_options = self.cfg.get("num_options", 4)
        self.target_update_frequency = self.cfg.get("target_update_frequency", 1000)
        self.start_learning = self.cfg.get("start_learning", 1000)
        self.gradient_clipping = self.cfg.get("gradient_clipping", 1.0)

        # Model
        self.model = self.models.get("model")
        self.target_model = self.models.get("target_model", None)

        if self.target_model is None:
            # Create a target model if not provided
            self.target_model = self.model.clone()
            self.models["target_model"] = self.target_model

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Internal variables
        self.current_option = None
        self.timesteps = 0

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent (called once before training)"""
        super().init(trainer_cfg=trainer_cfg)
        self.current_option = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Compute the actions for the given states

        :param states: Observations/states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total timesteps
        :type timesteps: int
        :return: Actions to be taken
        :rtype: torch.Tensor
        """
        states = states.to(self.device)

        with torch.no_grad():
            # Compute model outputs
            option_logits, termination_logits, action_logits, _ = self.model(states)

            # Option selection
            if self.current_option is None or self.should_terminate(termination_logits):
                # Select a new option
                option_probs = torch.softmax(option_logits, dim=-1)
                option_dist = torch.distributions.Categorical(option_probs)
                self.current_option = option_dist.sample()

            # Intra-option policy action selection
            action_logits = action_logits[:, self.current_option, :]
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

        return action.unsqueeze(-1).cpu()

    def should_terminate(self, termination_logits: torch.Tensor) -> bool:
        """Determine whether to terminate the current option

        :param termination_logits: Termination logits from the model
        :type termination_logits: torch.Tensor
        :return: True if the option should terminate, False otherwise
        :rtype: bool
        """
        termination_probs = torch.sigmoid(termination_logits)
        termination_prob = termination_probs[:, self.current_option]
        termination_dist = torch.distributions.Bernoulli(termination_prob)
        terminate = termination_dist.sample()
        return terminate.item() == 1

    def record_transition(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor,
                          rewards: torch.Tensor, 
                          next_states: torch.Tensor,
                          dones: torch.Tensor, 
                          infos:  Any) -> None:
        """Record experience in memory

        :param states: Current states
        :type states: torch.Tensor
        :param actions: Actions taken
        :type actions: torch.Tensor
        :param rewards: Rewards received
        :type rewards: torch.Tensor
        :param next_states: Next states
        :type next_states: torch.Tensor
        :param dones: Done flags
        :type dones: torch.Tensor
        :param infos: Additional information
        :type infos: list of dict
        """
        # Store the transition in memory
        self.memory.add({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "options": self.current_option.repeat(len(states))
        })

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Pre-interaction step (called before environment interaction)

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total timesteps
        :type timesteps: int
        """
        super().pre_interaction(timestep, timesteps)
        # Additional steps if needed

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Post-interaction step (called after environment interaction)

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total timesteps
        :type timesteps: int
        """
        super().post_interaction(timestep, timesteps)
        self.timesteps += 1

        # Learn if enough experience is gathered
        if self.timesteps > self.start_learning:
            self.learn()

        # Update target networks
        if self.timesteps % self.target_update_frequency == 0:
            self.update_target_networks()

    def learn(self) -> None:
        """Update the agent's networks"""
        batch = self.memory.sample(self.batch_size)

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        options = batch["options"].to(self.device)

        # Current model outputs
        option_logits, termination_logits, action_logits, q_values = self.model(states)
        action_logits = action_logits[range(len(options)), options]
        q_values = q_values[range(len(options)), options]

        # Next state outputs from target model
        with torch.no_grad():
            _, next_termination_logits, _, next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.max(dim=1)[0]
            termination_probs = torch.sigmoid(next_termination_logits)
            termination_probs = termination_probs[range(len(options)), options]

        # Compute targets
        continuation = (1 - termination_probs) * (1 - dones)
        targets = rewards + self.discount_factor * continuation * next_q_values

        # Critic loss
        critic_loss = nn.functional.mse_loss(q_values, targets)

        # Intra-option policy loss
        action_probs = torch.softmax(action_logits, dim=-1)
        action_log_probs = torch.log(torch.gather(action_probs, 1, actions.long()))
        advantages = (targets - q_values).detach()
        policy_loss = -torch.mean(action_log_probs * advantages)

        # Entropy regularization
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()
        policy_loss -= self.entropy_coefficient * entropy

        # Termination loss
        termination_logits = termination_logits[range(len(options)), options]
        termination_probs = torch.sigmoid(termination_logits)
        termination_target = (q_values - next_q_values).detach()
        termination_loss = torch.mean(termination_probs * termination_target)

        # Total loss
        total_loss = critic_loss + policy_loss + termination_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def update_target_networks(self) -> None:
        """Update target networks"""
        self.target_model.load_state_dict(self.model.state_dict())




