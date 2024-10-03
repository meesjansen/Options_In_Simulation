import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, TabularMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.q_learning import Q_LEARNING_DEFAULT_CONFIG

from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from my_agents.q_learning import Q_LEARNING


# Seed for reproducibility
seed = set_seed()  # e.g. `set_seed(42)` for fixed seed

# Define the models (stochastic and deterministic models) for the agent using helper mixin.
# - Policy: takes as input the environment's observation/state and returns an action

# define the model
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        self.epsilon = epsilon
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions), dtype=torch.float32)

    def compute(self, inputs, role):
        states = inputs["states"]
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), states.long()],
                               dim=-1, keepdim=True).view(-1,1)

        indexes = (torch.rand(states.shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {}
    

# instance VecEnvBase and setup task
headless = True  # set headless to False for rendering
env = get_env_instance(headless=headless, enable_livestream=True, enable_viewport=True)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from Rooms_Envs import ReachingFoodTask, TASK_CFG

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 4


sim_config = SimConfig(TASK_CFG)
task = ReachingFoodTask(name="ReachingFood", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)


# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device

# instantiate the model (assumes there is a wrapped environment: env)
# https://skrl.readthedocs.io/en/latest/api/models/tabular.html
models_q = {}
models_q["policy"] = EpilonGreedyPolicy(observation_space=env.observation_space,
                            action_space=env.action_space,
                            device=env.device,
                            num_envs=env.num_envs,
                            epsilon=0.15)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html

# cfg_agent["<KEY>"] = ...
Q_LEARNING_DEFAULT_CONFIG = {
    "discount_factor": 0.99,        # discount factor (gamma)

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "learning_rate": 0.5,           # learning rate (alpha)

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "experiment": {
        "directory": "./my_runs",            # experiment's parent directory
        "experiment_name": "Terrains_Env_Q_learning",      # experiment name
        "write_interval": 32,   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 250,      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}

cfg_agent = Q_LEARNING_DEFAULT_CONFIG.copy()


agent = Q_LEARNING(models=models_q,
                   memory=None,
                   cfg=cfg_agent,
                   observation_space=env.observation_space,
                   action_space=env.action_space,
                   device=env.device)



# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 5000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()