import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

from my_models.categorical import CategoricalMixin
from my_agents.ddpg import DDPG
from my_trainers.sequential import SequentialTrainer

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
from my_envs.KAMMA_eval import TorqueDistributionTask, TASK_CFG
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

# Instantiate a memory as experience replay
memory = RandomMemory(memory_size=1_000_000, num_envs=env.num_envs, device=device, replacement=False)

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

    "random_timesteps": 10,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": False,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1.0,        # final scale for the noise
        "timesteps": 50000.0,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/KAMMA_g1d_g21_seed5_eval",
        "experiment_name": "KAMMA_g1d_g21_seed5_eval",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": True,
        "wandb_kwargs": {"project": "KAMMA",
                         "entity": "meesjansen-Delft Technical University",
                         "name": "KAMMA_g1d_g21_seed5_eval",
                         "tags": ["DDPG", "KAMMA", "r18", "o6", "torq"],
                         "dir": "/workspace/Options_In_Simulation/my_runs"}    
                    }
}

cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.00, device=device)
cfg["gradient_steps"] = 1
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.999
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4
cfg["critic_learning_rate"] = 1e-3
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 500000


agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)


# agent.load("./my_runs/PPOc_rooms_r15_vel/PPOc_rooms_r15_vel/checkpoints/agent_100000.pt")
agent.load("./my_runs/KAMMA_g1d_g21/KAMMA_g1d_g21/checkpoints/agent_400000.pt")

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.eval()