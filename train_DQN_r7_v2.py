import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from my_models.deterministic import Shape, deterministic_model
from my_trainers.sequential import SequentialTrainer
from my_agents.dqn import DQN




# Seed for reproducibility
seed = set_seed(42)  # e.g. `set_seed(42)` for fixed seed
   

# instance VecEnvBase and setup task
headless = True  # set headless to False for rendering
env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.DQN_terrain_r7 import ReachingTargetTask, TASK_CFG

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 16


sim_config = SimConfig(TASK_CFG)
task = ReachingTargetTask(name="ReachingTarget", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)


# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device

# instantiate a memory as experience replay
memory = RandomMemory(memory_size=300_000, num_envs=env.num_envs, device=device, replacement=False)

# instantiate the agent's models (function approximators) using the model instantiator utility.
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models


models = {}
models["q_network"] = deterministic_model(observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          device=device,
                                          clip_actions=False,
                                          input_shape=Shape.OBSERVATIONS,
                                          hiddens=[64, 64],
                                          hidden_activation=["relu", "relu"],
                                          output_shape=Shape.ACTIONS,
                                          output_activation=None,
                                          output_scale=1.0)
models["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                 action_space=env.action_space,
                                                 device=device,
                                                 clip_actions=False,
                                                 input_shape=Shape.OBSERVATIONS,
                                                 hiddens=[64, 64],
                                                 hidden_activation=["relu", "relu"],
                                                 output_shape=Shape.ACTIONS,
                                                 output_activation=None,
                                                 output_scale=1.0)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
DQN_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 32,               # training batch size

    "discount_factor": 0.995,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "learning_rate": 1e-4,          # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 10_000,          # random exploration steps
    "learning_starts": 1000,           # learning starts after this many steps

    "update_interval": 1,           # agent update interval
    "target_update_interval": 100,   # target network update interval

    "exploration": {
        "initial_epsilon": 1.0,       # initial epsilon for epsilon-greedy exploration
        "final_epsilon": 0.1,        # final epsilon for epsilon-greedy exploration
        "timesteps": 100_000,            # timesteps for epsilon-greedy decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/r7_v2",            # experiment's parent directory
        "experiment_name": "DQN_Terrain_Rooms_r7_v2_long",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": True,             # whether to use Weights & Biases
        "wandb_kwargs": {"project":     "RL-Terrain-Simulation",
                        "entity":       "meesjansen-Delft Technical University",
                        "name":         "DQN_Terrain_Rooms_r7_v2_long",
                        "tags":         ["DQN", "Rooms"],
                        "dir":          "/workspace/Options_In_Simulation/my_runs"}    # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}

cfg = DQN_DEFAULT_CONFIG.copy()

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 50_000
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}


agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 300_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()