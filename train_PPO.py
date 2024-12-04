import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

from my_models.categorical import CategoricalMixin
from my_agents.ppo import PPO

# set the seed for reproducibility
seed = set_seed(42)


# Define the models (stochastic and deterministic models) for the agent using helper mixin.
# - Policy: takes as input the environment's observation/state and returns action probs
# - Value: takes the state as input and provides a state value to guide the policy
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)  # Output logits for discrete actions
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# instantiate and configure the task
headless = True  # set headless to False for rendering

env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.PPOd_terrain import ReachingTargetTask, TASK_CFG

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 16


sim_config = SimConfig(TASK_CFG)
task = ReachingTargetTask(name="ReachingTarget", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)


# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device

# instantiate a memory as experience replay 202
memory = RandomMemory(memory_size=20, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

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

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/PPO",            # experiment's parent directory
        "experiment_name": "PPO_r7",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": True,             # whether to use Weights & Biases
        "wandb_kwargs": {"project":     "RL-Terrain-Simulation",
                        "entity":       "meesjansen-Delft Technical University",
                        "name":         "PPO_Rooms_r7",
                        "tags":         ["PPO", "Rooms"],
                        "dir":          "/workspace/Options_In_Simulation/my_runs"}    # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
                    }          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 20
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 8
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 5e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0 # cfg_ppo["rollouts"] * env.num_envs * 4
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints
cfg_ppo["experiment"]["write_interval"] = 500
cfg_ppo["experiment"]["checkpoint_interval"] = 50_000


agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 250000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()