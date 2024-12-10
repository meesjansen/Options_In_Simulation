import gym
import torch
import torch.nn as nn
import numpy as np


from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.utils import set_seed

from my_models.deterministic import Shape
from my_agents.option_critic import *

# Seed for reproducibility
seed = set_seed(42)  # e.g. `set_seed(42)` for fixed seed


###############################################################################
# Network Definitions
###############################################################################
class MasterPolicyNetwork(nn.Module):
    """
    Master policy network that outputs a discrete distribution over options given the state.
    For simplicity, it outputs logits for each option.
    """
    def __init__(self, state_size, num_options, hidden_size=256):
        super(MasterPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_options)

        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc_out.weight, np.sqrt(0.01))

    def forward(self, states):
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        logits = self.fc_out(x)
        return logits  # [batch, num_options]


class OptionPolicyNetwork(nn.Module):
    """
    Option-specific policy network for continuous actions.
    This network outputs mean and log_std for the actions given the state and the active option.
    We will embed the option index into a one-hot vector and concatenate with the state.
    """
    def __init__(self, state_size, action_size, num_options, hidden_size=256):
        super(OptionPolicyNetwork, self).__init__()
        self.num_options = num_options
        # Input: state + one-hot option
        self.fc1 = nn.Linear(state_size + num_options, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))  # option-shared log_std

        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc_mean.weight, np.sqrt(0.01))

    def forward(self, states, options):
        # states: [batch, state_size]
        # options: [batch] containing integer option indices
        # Convert options to one-hot
        oh_options = torch.zeros((options.shape[0], self.num_options), device=states.device)
        oh_options[torch.arange(options.shape[0]), options] = 1.0
        x = torch.cat([states, oh_options], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class OptionValueNetwork(nn.Module):
    """
    Option-value network that outputs V(s, o) for each option.
    We compute a value for each option given the state and the option.
    """
    def __init__(self, state_size, num_options, hidden_size=256):
        super(OptionValueNetwork, self).__init__()
        self.num_options = num_options
        self.fc1 = nn.Linear(state_size + num_options, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc_out.weight, np.sqrt(0.01))

    def forward(self, states, options):
        oh_options = torch.zeros((options.shape[0], self.num_options), device=states.device)
        oh_options[torch.arange(options.shape[0]), options] = 1.0
        x = torch.cat([states, oh_options], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.fc_out(x)
        return v, None, None


class OptionTerminationNetwork(nn.Module):
    """
    Termination function network that outputs the probability of terminating an option.
    We model P(terminating | s, o).
    """
    def __init__(self, state_size, num_options, hidden_size=256):
        super(OptionTerminationNetwork, self).__init__()
        self.num_options = num_options
        self.fc1 = nn.Linear(state_size + num_options, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.fc1.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(2))
        nn.init.orthogonal_(self.fc_out.weight, np.sqrt(0.01))

    def forward(self, states, options):
        oh_options = torch.zeros((options.shape[0], self.num_options), device=states.device)
        oh_options[torch.arange(options.shape[0]), options] = 1.0
        x = torch.cat([states, oh_options], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc_out(x)
        termination_prob = torch.sigmoid(logits)
        return termination_prob, None, None




###############################################################################
# Running the training
###############################################################################
    
# instance VecEnvBase and setup task
headless = True  # set headless to False for rendering
env = get_env_instance(headless=headless, enable_livestream=True, enable_viewport=True)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.DQN_terrain import ReachingTargetTask, TASK_CFG

sim_config = SimConfig(TASK_CFG)
task = ReachingTargetTask(name="ReachingTarget", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 1


# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device

num_options = 4

# Create memory
memory = RandomMemory(memory_size=20, num_envs=1)

# Initialize models
master_policy = MasterPolicyNetwork(env.observation_space, num_options)
option_policy = OptionPolicyNetwork(env.observation_space, env.action_space, num_options)
option_value = OptionValueNetwork(env.observation_space, num_options)
option_termination = OptionTerminationNetwork(env.observation_space, num_options)

models = {
    "master_policy": master_policy,
    "option_policy": option_policy,
    "option_value": option_value,
    "option_termination": option_termination
}


# Agent configuration
PROXIMAL_POLICY_OPTION_CRITIC_DEFAULT_CONFIG = {
    "rollouts": 20,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": KLAdaptiveRL,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {{"kl_threshold": 0.008}},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    
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


# Create the agent
cfg_ppoc = PROXIMAL_POLICY_OPTION_CRITIC_DEFAULT_CONFIG.copy()
agent = OptionCriticAgent(models=models, memory=memory, cfg=cfg_ppoc)

# Configure and instantiate the trainer
cfg_trainer = {
    "timesteps": 20000,
    "headless": True
}

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start training
trainer.train()




