import torch
import torch.nn as nn
from typing import Tuple, List, Union

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.memories.torch import Memory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

from my_models.categorical import CategoricalMixin
from my_agents.ddpg import DDPG
from my_trainers.sequential_KA import SequentialTrainer

# set the seed for reproducibility
seed = set_seed(42)

# Define the models (stochastic and deterministic) for the agent using helper mixin.
# - Policy: takes as input the environment's observation/state and returns action
# - Value: takes the state as input and provides a state value to guide the policy
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions),
                                 nn.Sigmoid())

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
    
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

# Instantiate and configure the task
headless = True  # set headless to False for rendering

env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.KA_DDPG_1D import TorqueDistributionTask, TASK_CFG
from argparse import ArgumentParser 

arg_parser = ArgumentParser()
arg_parser.add_argument("--stiffness", type=float, default=0.035)
arg_parser.add_argument("--damping", type=float, default=0.005)
arg_parser.add_argument("--static_friction", type=float, default=0.85)
arg_parser.add_argument("--dynamic_friction", type=float, default=0.85)
arg_parser.add_argument("--yaw_constant", type=float, default=0.5)
arg_parser.add_argument("--linear_x", type=float, default=[1., 2.0])

parsed_config = arg_parser.parse_args().__dict__

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 1

# control
TASK_CFG["task"]["env"]["control"]["stiffness"] = parsed_config["stiffness"]
TASK_CFG["task"]["env"]["control"]["damping"] = parsed_config["damping"]

# friction
TASK_CFG["task"]["sim"]["default_physics_material"]["static_friction"] = parsed_config["static_friction"]
TASK_CFG["task"]["sim"]["default_physics_material"]["dynamic_friction"] = parsed_config["dynamic_friction"]

# commands
TASK_CFG["task"]["env"]["randomCommandVelocityRanges"]["yaw_constant"] = parsed_config["yaw_constant"]
TASK_CFG["task"]["env"]["randomCommandVelocityRanges"]["linear_x"] = parsed_config["linear_x"]


sim_config = SimConfig(TASK_CFG)
task = TorqueDistributionTask(name="TorqueDistribution", sim_config=sim_config, env=env)

env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

# Wrap the environment
env = wrap_env(env, "omniverse-isaacgym")
device = env.device

class FIFOMemory(Memory):
    def __init__(self, *args, replacement: bool = False, **kwargs):
        self.replacement = replacement
        # Remove it from kwargs if accidentally passed as a kwarg
        kwargs.pop("replacement", None)
        super().__init__(*args, **kwargs)

    def sample(
        self, names: Tuple[str], batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from FIFO memory randomly (without replacement by default)

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                The sampled tensors will have the shape: (batch_size, data_size)
        :rtype: list of torch.Tensor list
        """
        size = len(self)

        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()

        if self.replacement:
            indexes = torch.randint(0, size, (batch_size,))
        else:
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]

        if sequence_length > 1:
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)

    def add_samples(self, **tensors: torch.Tensor) -> None:
        if not tensors:
            raise ValueError("No samples to be recorded")

        tmp = tensors.get("states", tensors[next(iter(tensors))])
        dim, shape = tmp.ndim, tmp.shape

        if dim > 1 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)
            self.memory_index = (self.memory_index + 1) % self.memory_size
            if self.memory_index == 0:
                self.filled = True

        elif dim > 1 and shape[0] < self.num_envs:
            batch_size = shape[0]
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index:self.env_index + batch_size].copy_(tensor)
            self.env_index += batch_size

            if self.env_index >= self.num_envs:
                self.env_index = 0
                self.memory_index = (self.memory_index + 1) % self.memory_size
                if self.memory_index == 0:
                    self.filled = True

        elif dim > 1 and self.num_envs == 1:
            num_samples = shape[0]
            for name, tensor in tensors.items():
                if name in self.tensors:
                    insert_pos = self.memory_index
                    overflow = max(0, insert_pos + num_samples - self.memory_size)
                    if overflow == 0:
                        self.tensors[name][insert_pos:insert_pos + num_samples].copy_(tensor.unsqueeze(1))
                        self.memory_index += num_samples
                    else:
                        split = num_samples - overflow
                        self.tensors[name][insert_pos:insert_pos + split].copy_(tensor[:split].unsqueeze(1))
                        self.tensors[name][:overflow].copy_(tensor[split:].unsqueeze(1))
                        self.memory_index = overflow
                        self.filled = True
                    self.memory_index %= self.memory_size

        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1

            if self.env_index >= self.num_envs:
                self.env_index = 0
                self.memory_index = (self.memory_index + 1) % self.memory_size
                if self.memory_index == 0:
                    self.filled = True

        else:
            raise ValueError(f"Unexpected tensor shape: {shape}")
            # if not allow_overwrite and full → drop sample silently

# Instantiate a memory as experience replay
memory = FIFOMemory(memory_size=35_000, num_envs=env.num_envs, device=device, replacement=False)   # FIFO behaviour

# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)


# Configure PPO agent hyperparameters.
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 4096,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.01,                # soft update hyperparameter (tau)

    "actor_learning_rate": 3e-4,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": RunningStandardScaler,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.05, device=device),              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-4,        # final scale for the noise
        "timesteps": 700000.0,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/KA-DDPG_1D_g1",
        "experiment_name": "KA-DDPG_1D_g1",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": True,
        "wandb_kwargs": {"project": "KA-DDPG Dimension Study",
                         "entity": "meesjansen-Delft Technical University",
                         "name": "KA-DDPG_1D_g1",
                         "tags": ["DDPG", "KA", "o4", "torq"],
                         "dir": "/workspace/Options_In_Simulation/my_runs"}    
                    }
}

cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.999
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4
cfg["critic_learning_rate"] = 1e-3
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 700000


agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

task.memory = memory

# Configure and instantiate the RL trainer.
cfg_trainer = {"timesteps": 700000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()
