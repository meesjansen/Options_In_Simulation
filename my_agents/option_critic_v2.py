from typing import Union, Tuple, Dict, Any, Optional, Mapping

import copy
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.optim as optim

from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.agents.torch import Agent






OPTION_CRITIC_DEFAULT_CONFIG = {
    "discount_factor": 0.99,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "update_interval": 1,
    "target_update_interval": 100,
    "epsilon_start": 1.0,
    "epsilon_final": 0.1,
    "epsilon_decay": 10000,
    "temperature": 1.0,
    "gradient_clip": 1.0,
    "device": "cpu",
    "experiment": {
        "directory": "",
        "experiment_name": "OptionCritic",
        "write_interval": 100,
        "checkpoint_interval": 1000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {}
    }
}



class OptionCriticAgent(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Option-Critic Agent
        
        :param models: Dictionary containing the "policy" model.
        :param memory: Experience replay memory.
        :param observation_space: Observation space.
        :param action_space: Action space.
        :param device: Device for computations.
        :param cfg: Configuration dictionary.
        """
        _cfg = copy.deepcopy(OPTION_CRITIC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # Retrieve the policy model
        self.policy = self.models.get("policy")
        if self.policy is None:
            raise ValueError("The 'policy' model is required for the OptionCriticAgent.")

        # Move the model to the specified device
        self.policy.to(self.device)

        # Add the policy model to checkpoint modules
        self.checkpoint_modules["policy"] = self.policy

        # Parse configurations
        self.discount_factor = self.cfg["discount_factor"]
        self.learning_rate = self.cfg["learning_rate"]
        self.batch_size = self.cfg["batch_size"]
        self.update_interval = self.cfg["update_interval"]
        self.target_update_interval = self.cfg["target_update_interval"]
        self.epsilon_start = self.cfg["epsilon_start"]
        self.epsilon_final = self.cfg["epsilon_final"]
        self.epsilon_decay = self.cfg["epsilon_decay"]
        self.temperature = self.cfg["temperature"]
        self.gradient_clip = self.cfg["gradient_clip"]

        # Initialize epsilon for exploration
        self.epsilon = self.epsilon_start
        self.num_updates = 0

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Target network for stability
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.to(self.device)
        self.checkpoint_modules["target_policy"] = self.target_policy

        # Loss function
        self.mse_loss = nn.MSELoss()

        # Initialize temporary variables
        self._current_option = None
        self._current_option_terminated = True

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("train")

        # Ensure that the memory has the necessary tensors
        if self.memory is not None:
            self.memory.create_tensor("options", torch.long)
            self.memory.create_tensor("logits_options", torch.float)
            self.memory.create_tensor("terminations", torch.float)
            self.memory.create_tensor("beta", torch.float)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        # Prepare inputs
        inputs = {"states": states.to(self.device)}

        # Call the policy's act method
        outputs = self.policy.act(inputs, role="policy")

        # Retrieve actions and options
        actions = outputs["actions"]
        options = outputs["option"]

        # Store the current option for use in training
        self._current_option = options

        return actions

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
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        # Record agent-specific data
        if self.memory is not None:
            # Store the current option and termination flag
            options = self._current_option.to(self.memory.device)
            self.memory.add_samples({"states": states,
                                     "actions": actions,
                                     "rewards": rewards,
                                     "next_states": next_states,
                                     "terminated": terminated,
                                     "truncated": truncated,
                                     "options": options})

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        # Optionally update the agent before interaction
        if timestep % self.update_interval == 0 and timestep > 0:
            self.update(timestep, timesteps)

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        # Optionally update the agent after interaction
        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_final, self.epsilon - (self.epsilon_start - self.epsilon_final) / self.epsilon_decay)

        # Update target network
        if timestep % self.target_update_interval == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        if self.memory is None or len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        samples = self.memory.sample(self.batch_size)
        states = samples["states"].to(self.device)
        actions = samples["actions"].to(self.device)
        rewards = samples["rewards"].to(self.device)
        next_states = samples["next_states"].to(self.device)
        terminated = samples["terminated"].to(self.device)
        options = samples["options"].to(self.device)

        # Compute loss components
        self.optimizer.zero_grad()

        # Compute current Q-values and terminations
        current_outputs = self.policy.compute({"states": states}, role="policy")
        q_options = current_outputs["q_options"]
        beta = current_outputs["beta"]
        logits_options = current_outputs["logits_options"]

        # Compute next Q-values using target network
        with torch.no_grad():
            next_outputs = self.target_policy.compute({"states": next_states}, role="target_policy")
            next_q_options = next_outputs["q_options"]
            next_beta = next_outputs["beta"]

        # Critic Loss (TD Error for Option-Value Function)
        batch_indices = torch.arange(self.batch_size, dtype=torch.long)
        q_o = q_options[batch_indices, options]

        # Compute target Q-values
        next_v = torch.logsumexp(next_q_options / self.temperature, dim=1) * self.temperature
        target_q = rewards.squeeze() + self.discount_factor * (1 - terminated.squeeze()) * (next_v)

        td_error = target_q - q_o
        critic_loss = td_error.pow(2).mean()

        # Actor Loss (Policy Gradient for Intra-Option Policies)
        # Compute log probabilities of actions under the intra-option policies
        logits = logits_options[batch_indices, options]
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs[batch_indices, actions.squeeze().long()]

        # Advantage estimate
        advantages = td_error.detach()
        actor_loss = -(action_log_probs * advantages).mean()

        # Termination Loss
        beta_o = beta[batch_indices, options]
        option_advantages = q_o - next_v.detach()
        termination_loss = (beta_o * option_advantages.detach()).mean()

        # Total Loss
        total_loss = critic_loss + actor_loss + termination_loss

        # Backpropagation
        total_loss.backward()

        # Gradient clipping
        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)

        # Optimizer step
        self.optimizer.step()

        # Logging
        self.track_data("Loss / Critic Loss", critic_loss.item())
        self.track_data("Loss / Actor Loss", actor_loss.item())
        self.track_data("Loss / Termination Loss", termination_loss.item())
        self.track_data("Loss / Total Loss", total_loss.item())
        self.track_data("Exploration / Epsilon", self.epsilon)

        self.num_updates += 1
