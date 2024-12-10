from typing import Union, Tuple, Dict, Any, Optional, Mapping

import copy
import math
import gym
import gymnasium
import numpy as np

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
PROXIMAL_POLICY_OPTION_CRITIC_DEFAULT_CONFIG = {
    "rollouts": 20,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": KLAdaptiveLR,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})


    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_reg_coefficient": 0.01,
    "termination_reg_coefficient": 0.01,

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "num_options": 5,
    "temperature": 1.0,

    
    "device": "gpu",

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
# [end-config-dict-torch]


class PPOC(Agent):
    """
    PPOC agent with a rollout parameter:
    - Collect exactly rollout_length steps of experience.
    - After rollout_length steps are collected, compute advantages, returns, and run PPO-like updates.
    - Clear memory and repeat.

    This structure mirrors the standard PPO training cycle, but applied to PPOC.
    """
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        _cfg = copy.deepcopy(PROXIMAL_POLICY_OPTION_CRITIC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)


        self.num_options = self.cfg["num_options"]
        self.clip_ratio = self.cfg["ratio_clip"]
        self.lr = self.cfg["learning_rate"]
        self.gamma = self.cfg["discount_factor"]
        self.lam = self.cfg["lambda"]
        self.epochs = self.cfg["learning_epochs"]
        self.mini_batches = self.cfg["mini_batches"]
        self.entropy_coef = self.cfg["entropy_reg_coefficient"]
        self.termination_reg = self.cfg["termination_reg_coefficient"]
        self._rollouts = self.cfg["rollouts"]
        self._learning_starts = self.cfg["learning_starts"]

        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.value_clip = self.cfg["value_clip"]
        self.clip_predicted_values = self.cfg["clip_predicted_values"]
        self.temperature = self.cfg["temperature"]
        self._kl_threshold = self.cfg["kl_threshold"]

    
        self.device = self.cfg["device"]
        self.experiment = self.cfg["experiment"]
        self.random_timesteps = self.cfg["random_timesteps"]

        self.state_preprocessor = self.cfg["state_preprocessor"]
        self.state_preprocessor_kwargs = self.cfg["state_preprocessor_kwargs"]
        self.value_preprocessor = self.cfg["value_preprocessor"]
        self.value_preprocessor_kwargs = self.cfg["value_preprocessor_kwargs"]

        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]
        self._learning_rate_scheduler_kwargs = self.cfg["learning_rate_scheduler_kwargs"]


        self.master_policy = models["master_policy"].to(self.device)
        self.option_policy = models["option_policy"].to(self.device)
        self.option_value = models["option_value"].to(self.device)
        self.option_termination = models["option_termination"].to(self.device)

        # checkpoint models
        self.checkpoint_modules["master_policy"] = self.master_policy
        self.checkpoint_modules["option_policy"] = self.option_policy
        self.checkpoint_modules["option_value"] = self.option_value
        self.checkpoint_modules["option_termination"] = self.option_termination


        self.optimizer = torch.optim.Adam(
            list(self.master_policy.parameters()) +
            list(self.option_policy.parameters()) +
            list(self.option_value.parameters()) +
            list(self.option_termination.parameters()),
            lr=self.lr
        )

        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

        self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor


        self.current_option = None
        self._rollout = 0


    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="action_log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="option_log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="current_option", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="entropy", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)


            # tensors sampled during training
            self._tensors_names = ["states", "actions", "rewards", "terminated", "action_log_prob", "option_log_prob", "values", "current_option", "entropy"]
        
        # create temporary variables needed for storage and computation
        self._current_action_log_prob = None
        self._current_option_log_prob = None
        self._current_next_states = None



    def act(self, states, inference=False):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)

        # Determine if we need to select or terminate an option
        if self.current_option is None:
            with torch.no_grad():
                logits = self.master_policy(states)
                dist = torch.distributions.Categorical(logits=logits)
                self.current_option = dist.sample()
                option_log_prob = dist.log_prob(self.current_option)
        else:
            # Check if option terminates
            with torch.no_grad():
                term_prob = self.option_termination(states, self.current_option)
                terminated = (torch.rand_like(term_prob) < term_prob).squeeze(-1)
                if terminated.item():
                    logits = self.master_policy(states)
                    dist = torch.distributions.Categorical(logits=logits)
                    self.current_option = dist.sample()
                    option_log_prob = dist.log_prob(self.current_option)
                else:
                    option_log_prob = torch.tensor([0.0], device=self.device)  # no new option selected

        # Sample action from option policy
        with torch.no_grad():
            mean, log_std = self.option_policy(states, self.current_option)
            dist = torch.distributions.Normal(mean, log_std.exp())
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)

        self._current_action_log_prob = action_log_prob
        self._current_option_log_prob = option_log_prob

        return action, action_log_prob, None
    


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

        if self.memory is not None:
            self._current_next_states = next_states

            # compute values
            values, _, _ = self.option_value({"states": self._state_preprocessor(states)}, self.current_option)
            values = self._value_preprocessor(values, inverse=True)

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_action_log_prob, values=values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_action_log_prob, values=values)



    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass



    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)



    def compute_advantages_and_returns(self):

        # fetch tensors from memory
        rewards = self.memory.get_tensor_by_name("rewards")
        terminated = self.memory.get_tensor_by_name("terminated").float()  # convert bool to float for calculations
        values = self.memory.get_tensor_by_name("values")


        next_values = torch.zeros_like(values)
    
        with torch.no_grad():
            # The last transition in memory has a corresponding next state that we can use to compute next_values
            # We rely on the memory interface to fetch the last recorded next state:
            last_next_states = self.memory.get_tensor_by_name("next_states", idx=[-1])
            last_options = self.memory.get_tensor_by_name("current_option", idx=[-1]).long()
            # Preprocess the last next states if necessary
            last_next_states = self._state_preprocessor(last_next_states)
            v, _, _ = self.option_value({"states": last_next_states}, last_options)
            v = self._value_preprocessor(v, inverse=True)
            next_values[-1] = v

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        # GAE calculation backwards
        for t in reversed(range(rewards.shape[0])):
            next_val = next_values[t] if t == rewards.shape[0] - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * (1.0 - terminated[t]) * next_val - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - terminated[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        return advantages, returns
    


    def _update(self):
        """
        Perform PPOC updates after collecting a full rollout of rollout_length steps.
        """
        advantages, returns = self.compute_advantages_and_returns()

        values = self.memory.get_tensor_by_name("values")
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)


        # fetch all relevant data from memory
        states = self.memory.get_tensor_by_name("states")
        actions = self.memory.get_tensor_by_name("actions")
        action_log_prob = self.memory.get_tensor_by_name("action_log_prob")
        option_log_prob = self.memory.get_tensor_by_name("option_log_prob")
        options = self.memory.get_tensor_by_name("current_option").long()
        advantages = self.memory.get_tensor_by_name("advantages")
        returns = self.memory.get_tensor_by_name("returns")
        entropy = self.memory.get_tensor_by_name("entropy")

        cumulative_surr_action = 0.0
        cumulative_surr_option = 0.0
        cumulative_value_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_termination_loss = 0.0

        # number of learning epochs and mini-batches
        for epoch in range(self.epochs):
            # sample all transitions and divide into mini-batches
            kl_divergences = []
            sampler = self.memory.sample_all(self.mini_batches)
            for indices in sampler:
                s_mb = states[indices].to(self.device)
                a_mb = actions[indices].to(self.device)
                old_lp_mb = action_log_prob[indices].to(self.device)
                opt_lp_mb = option_log_prob[indices].to(self.device)
                adv_mb = advantages[indices].to(self.device)
                ret_mb = returns[indices].to(self.device)
                opt_mb = options[indices].to(self.device)

                # master policy distribution
                logits, _, _ = self.master_policy({"states": self._state_preprocessor(s_mb)})
                master_dist = torch.distributions.Categorical(logits=logits)
                new_opt_lp = master_dist.log_prob(opt_mb.squeeze(-1))

                # option policy distribution
                mean, log_std, _ = self.option_policy({"states": self._state_preprocessor(s_mb)}, opt_mb)
                dist = torch.distributions.Normal(mean, log_std.exp())
                new_lp = dist.log_prob(a_mb).sum(-1, keepdim=True)

                # compute ratio for PPO action loss
                ratio = torch.exp(new_lp - old_lp_mb)
                surr_action = torch.min(ratio * adv_mb,
                                        torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = new_lp - old_lp_mb
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)
                
                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break


                # master policy loss
                ratio_opt = torch.exp(new_opt_lp - opt_lp_mb)
                surr_option = torch.min(ratio_opt * adv_mb,
                                        torch.clamp(ratio_opt, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_mb)

                # value loss MSE
                v, _, _ = self.option_value({"states": self._state_preprocessor(s_mb)}, opt_mb)
                v = self._value_preprocessor(v, inverse=True)
                value_loss = 0.5 * (ret_mb - v).pow(2).mean()

                # entropy terms
                option_entropy = dist.entropy().sum(-1).mean()
                master_entropy = master_dist.entropy().mean()
                total_entropy = option_entropy + master_entropy

                # termination probability regularization
                term_prob, _, _ = self.option_termination({"states": self._state_preprocessor(s_mb)}, opt_mb)
                # Note: if option_termination model returns only probabilities, adapt accordingly
                termination_loss = term_prob.mean() * self.termination_reg

                loss = - (surr_action.mean() + surr_option.mean()) \
                       + value_loss \
                       - self.entropy_coef * (option_entropy + master_entropy) \
                       + termination_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_norm_clip)
                self.optimizer.step()

                # Update cumulative losses
                cumulative_surr_action += surr_action.mean().item()
                cumulative_surr_option += surr_option.mean().item()
                cumulative_value_loss += value_loss.item()
                cumulative_entropy_loss += total_entropy.item()
                cumulative_termination_loss += termination_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    self.scheduler.step(torch.tensor(kl_divergences, device=self.device).mean())
                else:
                    self.scheduler.step()

        total_updates = self.epochs * self.mini_batches
        # After completing all epochs and mini-batches for this update:
        avg_surr_action = cumulative_surr_action / total_updates
        avg_surr_option = cumulative_surr_option / total_updates
        avg_value_loss = cumulative_value_loss / total_updates
        avg_entropy_loss = cumulative_entropy_loss / total_updates
        avg_termination_loss = cumulative_termination_loss / total_updates

        # Record data
        self.track_data("Loss / Surrogate action loss", -avg_surr_action)  # negative because we took -(surr_action)
        self.track_data("Loss / Surrogate option loss", -avg_surr_option)  # negative for consistency
        self.track_data("Loss / Value loss", avg_value_loss)
        self.track_data("Loss / Entropy loss", -self.entropy_coef * avg_entropy_loss) 
        self.track_data("Loss / Termination loss", avg_termination_loss)

        # If you have a policy standard deviation to track:
        # Assuming your option_policy distributions have a stddev method:
        if hasattr(self.option_policy, "distribution"):
            stddev = self.option_policy.distribution(role="policy").stddev.mean().item()
            self.track_data("Policy / Standard deviation", stddev)

        # If using a learning rate scheduler, track the current learning rate:
        if self._learning_rate_scheduler:
            if isinstance(self.scheduler, KLAdaptiveLR) and len(kl_divergences) > 0:
                self.scheduler.step(torch.tensor(kl_divergences, device=self.device).mean())
            else:
                self.scheduler.step()
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # clear memory after update
        self._rollout = 0
        self.current_option = None