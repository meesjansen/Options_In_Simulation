from typing import Union, Tuple, Dict, Any, Optional, Mapping

import copy
import math
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.agents.torch import Agent
from skrl.resources.schedulers.torch import KLAdaptiveLR


# [start-config-dict-torch]
OPTION_CRITIC_DEFAULT_CONFIG = {
    "learning_rate": 5e-3,

    "discount_factor": 0.99,

    "exploration": {
        "initial_epsilon": 1.0,       # initial epsilon for epsilon-greedy exploration
        "final_epsilon": 0.1,        # final epsilon for epsilon-greedy exploration
        "timesteps": 10000,            # timesteps for epsilon-greedy decay
    },

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "batch_size": 32,
    "update_interval": 4,
    "prime_update_interval": 100,

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "entropy_reg_coefficient": 0.01,
    "termination_reg_coefficient": 0.01,

    "num_options": 5,
    "temperature": 1.0,

    "gradient_clip": 1.0,
    "device": "gpu",
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
# [end-config-dict-torch]


class OptionCriticAgent(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Option-Critic (OC) agent implementation.
        
        :param models: Dictionary containing the "OC" model and prime model for DQN.
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

        # Retrieve the OC model and prime for Option selection
        self.OC_NN = self.models.get("Options-Critic")
        self.prime_NN = self.models.get("Prime Option-Critic")
               
        # Add the models to checkpoint modules
        self.checkpoint_modules["Option-Critic"] = self.OC_NN
        self.checkpoint_modules["Prime Option-Critic"] = self.prime_NN

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.OC_NN is not None:
                self.OC_NN.broadcast_parameters()
                if self.prime_NN is not None and self.policy is not self.prime_NN:
                    self.prime_NN.broadcast_parameters()
        
        # Parse configurations
        self.learning_rate = self.cfg["learning_rate"]

        self.discount_factor = self.cfg["discount_factor"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self.batch_size = self.cfg["batch_size"]
        self.update_interval = self.cfg["update_interval"]
        self.prime_update_interval = self.cfg["prime_update_interval"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self.entropy_reg_coefficient = self.cfg["entropy_reg_coefficient"]
        self.termination_reg_coefficient = self.cfg["termination_reg_coefficient"]

        self.num_options = self.cfg["num_options"]
        self.temperature = self.cfg["temperature"]

        self.gradient_clip = self.cfg["gradient_clip"]
        self.device = self.cfg["device"]

        # Initialize epsilon for exploration
        self.epsilon = self.epsilon_start
        self.num_updates = 0

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        # Initialize temporary variables
        self._current_option = None
        self._current_option_terminated = True

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("train")

        # Ensure that the memory has the necessary tensors
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="options", size=1, dtype=torch.int32)
            self.memory.create_tensor(name="s_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="so_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages_DQN", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32) # check this for size
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="a_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages_termin", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="beta_termin", size=self.num_options, dtype=torch.float)


    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        

        states = self._state_preprocessor(states)

        if not self._exploration_timesteps:
            return torch.argmax(self.OC_NN.act({"states": states}, role="Option-Critic")[0], dim=1, keepdim=True), None, None

        # sample random actions
        actions = self.OC_NN.random_act({"states": states}, role="Option-Critic")[0]
        if timestep < self._random_timesteps:
            return actions, None, None

        # sample actions with epsilon-greedy policy
        epsilon = self._exploration_final_epsilon + (self._exploration_initial_epsilon - self._exploration_final_epsilon) \
                * math.exp(-1.0 * timestep / self._exploration_timesteps)

        indexes = (torch.rand(states.shape[0], device=self.device) >= epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.argmax(self.q_network.act({"states": states[indexes]}, role="q_network")[0], dim=1, keepdim=True)

        
        # record epsilonup
        self.track_data("Exploration / Exploration epsilon", epsilon)

        return actions, None, None


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
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        # Optionally update the agent after interaction
        if timestep % self.update_interval == 0 and timestep > 0:
            self._update(timestep, timesteps)
        
        # write tracking data and checkpoints
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
