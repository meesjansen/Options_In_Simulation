import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

from my_models.categorical import CategoricalMixin
from my_agents.ppo import PPO
from my_trainers.sequential import SequentialTrainer

# set the seed for reproducibility
seed = set_seed(42)

# Define the models (stochastic and deterministic) for the agent using helper mixin.
# - Policy: takes as input the environment's observation/state and returns action probabilities
# - Value: takes the state as input and provides a state value to guide the policy
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# Instantiate and configure the task
headless = True  # set headless to False for rendering

env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

print("0-Starting warm-start training using supervised learning (MSE loss) for multiple phases...")

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.PPOc_warm_r16_Forestry import ReachingTargetTask, TASK_CFG
from argparse import ArgumentParser 

arg_parser = ArgumentParser()
arg_parser.add_argument("--linearVelocityXYRewardScale", type=float, default=1.0)
arg_parser.add_argument("--linearVelocityZRewardScale", type=float, default=-4.0)
arg_parser.add_argument("--angularVelocityXYRewardScale", type=float, default=-0.5)
arg_parser.add_argument("--actionRateRewardScale", type=float, default=-0.05)
arg_parser.add_argument("--fallenOverRewardScale", type=float, default=-200.0)
arg_parser.add_argument("--slipLongitudinalRewardScale", type=float, default=-5.0)
arg_parser.add_argument("--stiffness", type=float, default=0.05)
arg_parser.add_argument("--damping", type=float, default=0.005)
arg_parser.add_argument("--torq_constant", type=float, default=7.2)
arg_parser.add_argument("--torq_FF_gain", type=float, default=0.1)
arg_parser.add_argument("--static_friction", type=float, default=1.0)
arg_parser.add_argument("--dynamic_friction", type=float, default=1.0)
arg_parser.add_argument("--yaw_constant", type=float, default=0.5)
arg_parser.add_argument("--linear_x", type=float, default=1.0)

parsed_config = arg_parser.parse_args().__dict__

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 16

# sweep reward components
TASK_CFG["task"]["env"]["learn"]["linearVelocityXYRewardScale"] = parsed_config["linearVelocityXYRewardScale"]
TASK_CFG["task"]["env"]["learn"]["linearVelocityZRewardScale"] = parsed_config["linearVelocityZRewardScale"]
TASK_CFG["task"]["env"]["learn"]["angularVelocityXYRewardScale"] = parsed_config["angularVelocityXYRewardScale"]
TASK_CFG["task"]["env"]["learn"]["actionRateRewardScale"] = parsed_config["actionRateRewardScale"]
TASK_CFG["task"]["env"]["learn"]["fallenOverRewardScale"] = parsed_config["fallenOverRewardScale"]
TASK_CFG["task"]["env"]["learn"]["slipLongitudinalRewardScale"] = parsed_config["slipLongitudinalRewardScale"]

# control
TASK_CFG["task"]["env"]["control"]["stiffness"] = parsed_config["stiffness"]
TASK_CFG["task"]["env"]["control"]["damping"] = parsed_config["damping"]
TASK_CFG["task"]["env"]["control"]["torq_constant"] = parsed_config["torq_constant"]
TASK_CFG["task"]["env"]["control"]["torq_FF_gain"] = parsed_config["torq_FF_gain"]

# friction
TASK_CFG["task"]["sim"]["default_physics_material"]["static_friction"] = parsed_config["static_friction"]
TASK_CFG["task"]["sim"]["default_physics_material"]["dynamic_friction"] = parsed_config["dynamic_friction"]

# commands
TASK_CFG["task"]["env"]["randomCommandVelocityRanges"]["yaw_constant"] = parsed_config["yaw_constant"]
TASK_CFG["task"]["env"]["randomCommandVelocityRanges"]["linear_x"] = parsed_config["linear_x"]

print("1-Starting warm-start training using supervised learning (MSE loss) for multiple phases...")

sim_config = SimConfig(TASK_CFG)
task = ReachingTargetTask(name="ReachingTarget", sim_config=sim_config, env=env)
print("2-Starting warm-start trainitrain_rooms_r15.pyng using supervised learning (MSE loss) for multiple phases...")

env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)
print("3-Starting warm-start training using supervised learning (MSE loss) for multiple phases...")

# Wrap the environment
env = wrap_env(env, "omniverse-isaacgym")
device = env.device

# Instantiate a memory as experience replay
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device, replacement=False)

# Instantiate the agent's models.
models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

# Configure PPO agent hyperparameters.
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
        "directory": "/workspace/Options_In_Simulation/my_runs/PPOc_warm_r16_Forestry",
        "experiment_name": "PPOc_warm_r16_Forestry",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": True,
        "wandb_kwargs": {"project": "PPO_curriculum",
                         "entity": "meesjansen-Delft Technical University",
                         "name": "PPOc_warm_r16_Forestry",
                         "tags": ["PPOc", "warm-Curriculum", "r16", "o163", "torq", "Forestry"],
                         "dir": "/workspace/Options_In_Simulation/my_runs"}    
                    }
}

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 1024
cfg_ppo["learning_epochs"] = 5
cfg_ppo["mini_batches"] = 6
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 3e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0 
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.001
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = None
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg_ppo["experiment"]["write_interval"] = 100
cfg_ppo["experiment"]["checkpoint_interval"] = 20_000

print("4-Starting warm-start training using supervised learning (MSE loss) for multiple phases...")


agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

print("5- Starting warm-start training using supervised learning (MSE loss) for multiple phases...")

###############################################################################
#                            Warm-Start Phase
#
# We now add five sequential phases:
# 1. Drive straight (all wheels same torque).
# 2. Rotate left (left wheels positive, right wheels negative).
# 3. Rotate right (left wheels negative, right wheels positive).
# 4. Drive in a small circle to the left (differential: left wheels lower).
# 5. Drive in a small circle to the right (differential: left wheels higher).
###############################################################################

# Define expert (heuristic) functions for each phase.
def expert_straight(num_envs, num_actions, device):
    # All wheels same torque (e.g., 0.5)
    return torch.full((num_envs, num_actions), 5.0, device=device)

def expert_rotate_left(num_envs, num_actions, device):
    # Order assumed: [front_left, front_right, rear_left, rear_right]
    actions = torch.zeros((num_envs, num_actions), device=device)
    actions[:, 0] = 5.0   # front_left positive
    actions[:, 1] = -5.0  # front_right negative
    actions[:, 2] = 5.0   # rear_left positive
    actions[:, 3] = -5.0  # rear_right negative
    return actions

def expert_rotate_right(num_envs, num_actions, device):
    actions = torch.zeros((num_envs, num_actions), device=device)
    actions[:, 0] = -5.0  # front_left negative
    actions[:, 1] = 5.0   # front_right positive
    actions[:, 2] = -5.0  # rear_left negative
    actions[:, 3] = 5.0   # rear_right positive
    return actions

def expert_circle_left(num_envs, num_actions, device):
    # Forward torque on all wheels, with left wheels slightly lower torque than right wheels.
    actions = torch.zeros((num_envs, num_actions), device=device)
    actions[:, 0] = 4.0  # front_left
    actions[:, 1] = 6.0  # front_right
    actions[:, 2] = 4.0  # rear_left
    actions[:, 3] = 6.0  # rear_right
    return actions

def expert_circle_right(num_envs, num_actions, device):
    # Forward torque on all wheels, with left wheels slightly higher torque than right wheels.
    actions = torch.zeros((num_envs, num_actions), device=device)
    actions[:, 0] = 6.0  # front_left
    actions[:, 1] = 4.0  # front_right
    actions[:, 2] = 6.0  # rear_left
    actions[:, 3] = 4.0  # rear_right
    return actions

# List of warm-start phases: (phase_name, expert_function, num_timesteps)
warm_start_phases = [
    ("straight", expert_straight, 100),
    ("rotate_left", expert_rotate_left, 100),
    ("rotate_right", expert_rotate_right, 100),
    ("circle_left", expert_circle_left, 100),
    ("circle_right", expert_circle_right, 100)
]

# # Enable warm-start mode in the environment.
# task.warm_start = True
# # Freeze the value network (only train the policy).
# for param in agent.models["value"].parameters():
#     param.requires_grad = False

# # Create an optimizer for the policy network only.
# warm_optimizer = torch.optim.Adam(agent.models["policy"].parameters(), lr=cfg_ppo["learning_rate"])
# mse_loss_fn = nn.MSELoss()

# # Reset the environment (Gym v0.26+ reset returns (obs, infos))
# obs, infos = env.reset()

# # Iterate over each warm-start phase.
# for phase_name, expert_fn, phase_steps in warm_start_phases:
#     print(f"--- Warm-start phase: {phase_name} ({phase_steps} timesteps) ---")
#     task.phase_name = phase_name
#     for step in range(phase_steps):
#         # Get the phase-specific expert actions.
#         expert_actions = expert_fn(env.num_envs, env.action_space.shape[0], device)
#         # Optionally, you could also set extras["expert_actions"] here if needed.

#         obs, reward, terminated, truncated, extras = env.step(expert_actions)
#         done = terminated | truncated
#         state = obs  # observation tensor

               
#         # Forward pass through the policy network.
#         policy_output, _, _ = agent.models["policy"].compute({"states": state}, role="policy")

#         # render scene
#         if not headless:
#             env.render()

#         loss = mse_loss_fn(policy_output, expert_actions)
#         warm_optimizer.zero_grad()
#         loss.backward()
#         warm_optimizer.step()
        
#         if step % 100 == 0:
#             print(f"Phase {phase_name} step {step}: MSE Loss = {loss.item():.4f}")
#         if done.any():
#             obs, infos = env.reset()
#     # End of phase.
#     print(f"Phase {phase_name} completed.")

# print("Warm-start training completed. Transitioning to PPO training...")
# # Unfreeze the value network.
# for param in agent.models["value"].parameters():
#     param.requires_grad = True
# # Disable warm-start mode.
# task.warm_start = False




# Configure and instantiate the RL trainer.
cfg_trainer = {"timesteps": 307_206, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start PPO training.
trainer.train()
