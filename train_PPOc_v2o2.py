import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
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


# Modified Policy and Value classes to incorporate CNN
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        
        self.num_observations = self.num_observations
        self.num_height_channels = 4
        self.height_size = 13
        self.num_height_features = self.height_size * self.height_size * self.num_height_channels
        self.num_other_features = self.num_observations - self.num_height_features

        # CNN for heightmap data
        self.cnn = nn.Sequential(
            nn.Conv2d(self.num_height_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # reduces 13x13 to 6x6
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU()
        )

        # MLP for other features + CNN output
        self.fc = nn.Sequential(
            nn.Linear(self.num_other_features + 128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = inputs["states"]
        # Split features into other + height
        other_features = x[:, :self.num_other_features]
        height_data = x[:, self.num_other_features:].reshape(-1, self.num_height_channels, self.height_size, self.height_size)

        height_features = self.cnn(height_data)
        combined = torch.cat([other_features, height_features], dim=-1)
        return self.fc(combined), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_observations = self.num_observations
        self.num_height_channels = 4
        self.height_size = 13
        self.num_height_features = self.height_size * self.height_size * self.num_height_channels
        self.num_other_features = self.num_observations - self.num_height_features

        # CNN same as above
        self.cnn = nn.Sequential(
            nn.Conv2d(self.num_height_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.num_other_features + 128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))
    def compute(self, inputs, role):
        x = inputs["states"]
        other_features = x[:, :self.num_other_features]
        height_data = x[:, self.num_other_features:].reshape(-1, self.num_height_channels, self.height_size, self.height_size)

        height_features = self.cnn(height_data)
        combined = torch.cat([other_features, height_features], dim=-1)
        return self.fc(combined), {}


# instantiate and configure the task
headless = True  # set headless to False for rendering
env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.PPOc_terrain_v2o2 import ReachingTargetTask, TASK_CFG

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 16

sim_config = SimConfig(TASK_CFG)
task = ReachingTargetTask(name="ReachingTarget", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

env = wrap_env(env, "omniverse-isaacgym")
device = env.device

# memory
memory = RandomMemory(memory_size=20, num_envs=env.num_envs, device=device, replacement=False)

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

PPO_DEFAULT_CONFIG = {
    "rollouts": 16,
    "learning_epochs": 8,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "lambda": 0.95,
    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": False,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 1.0,
    "kl_threshold": 0,
    "rewards_shaper": None,
    "time_limit_bootstrap": False,
    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/PPOc_v2o2",
        "experiment_name": "PPOc_v2o2",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": True,
        "wandb_kwargs": {
            "project": "RL-Terrain-Simulation",
            "entity": "meesjansen-Delft Technical University",
            "name": "PPOc_Rooms_v2o2",
            "tags": ["PPOc", "Rooms"],
            "dir": "/workspace/Options_In_Simulation/my_runs"
        }
    }
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
cfg_ppo["learning_starts"] = 0
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
cfg_ppo["experiment"]["write_interval"] = 500
cfg_ppo["experiment"]["checkpoint_interval"] = 50000

agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.train()