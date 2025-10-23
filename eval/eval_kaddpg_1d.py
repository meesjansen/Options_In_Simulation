import os
import re
from pathlib import Path

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

from my_agents.ddpg import DDPG
from my_trainers.sequential import SequentialTrainer

# ----------------------------
# Models
# ----------------------------
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
            nn.Sigmoid(),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        import torch as _torch
        return self.net(_torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

# ----------------------------
# Env / Task
# ----------------------------
headless = True  # set headless to False for rendering
env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.KA_DDPG_1D_eval import TorqueDistributionTask, TASK_CFG
from argparse import ArgumentParser

# ----------------------------
# Argparse (+ env fallbacks)
# ----------------------------
arg_parser = ArgumentParser(description="Evaluate KA-DDPG 1D agent")

# physics / control
arg_parser.add_argument("--stiffness", type=float, default=0.035)
arg_parser.add_argument("--damping", type=float, default=0.005)
arg_parser.add_argument("--static_friction", type=float, default=0.85)
arg_parser.add_argument("--dynamic_friction", type=float, default=0.85)
arg_parser.add_argument("--yaw_constant", type=float, default=0.5)
arg_parser.add_argument("--linear_x", type=float, nargs=2, default=[1.0, 2.0])

# eval control
arg_parser.add_argument("--seed", type=int, default=None, help="Evaluation RNG seed")
arg_parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path")
arg_parser.add_argument("--experiment-name", type=str, default=None, help="Explicit experiment name")

# (optional) train-run reconstruction knobs
arg_parser.add_argument("--train-seed", type=int, default=None, help="Training seed to identify the run")
arg_parser.add_argument("--fifo", choices=["fifo", "nofifo"], default=None, help="fifo/nofifo used in TRAIN")
arg_parser.add_argument("--curriculum", type=str, default=None, help="curriculum used in TRAIN (e.g., random)")
arg_parser.add_argument("--strategy", type=str, default=None, help="strategy label used in TRAIN (e.g., RLIL)")
arg_parser.add_argument("--checkpoint-step", type=int, default=None, help="Checkpoint step (e.g., 500000)")
arg_parser.add_argument("--root", type=str, default="/workspace/Options_In_Simulation", help="Project root")

parsed_config = vars(arg_parser.parse_args())

# fallbacks from environment (router may set these)
if parsed_config["seed"] is None and os.getenv("EVAL_SEED"):
    parsed_config["seed"] = int(os.getenv("EVAL_SEED"))
if parsed_config["checkpoint"] is None and os.getenv("EVAL_CHECKPOINT"):
    parsed_config["checkpoint"] = os.getenv("EVAL_CHECKPOINT")
if parsed_config["experiment_name"] is None and os.getenv("EVAL_EXPERIMENT_NAME"):
    parsed_config["experiment_name"] = os.getenv("EVAL_EXPERIMENT_NAME")

# ----------------------------
# Seed & TASK config
# ----------------------------
seed = set_seed(parsed_config["seed"] if parsed_config["seed"] is not None else 42)

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

# ----------------------------
# Memory & Models
# ----------------------------
memory = RandomMemory(memory_size=1_000_000, num_envs=env.num_envs, device=device, replacement=False)

models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)

# ----------------------------
# DDPG config (with experiment wiring)
# ----------------------------
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,
    "batch_size": 4096,
    "discount_factor": 0.99,
    "polyak": 0.01,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
    "random_timesteps": 10,
    "learning_starts": 0,
    "grad_norm_clip": 0,
    "exploration": {
        "noise": False,
        "initial_scale": 1.0,
        "final_scale": 1.0,
        "timesteps": 50_000.0,
    },
    "rewards_shaper": None,
    "mixed_precision": False,
    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/eval_kaddpg_1d",  # base dir for eval logs
        "experiment_name": None,   # filled below
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": True,
        "wandb_kwargs": {
            "project": "KA-DDPG Dimension Study",
            "entity": "meesjansen-Delft Technical University",
            "name": None,  # filled below
            "tags": ["DDPG", "KAMMA", "o6", "torq"],
            "dir": "/workspace/Options_In_Simulation/my_runs",
        },
    },
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
cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 500_000

# ----------------------------
# Checkpoint & experiment-name resolution
# ----------------------------
algo = "kaddpg"
action_dim = "1d"

fifo = parsed_config.get("fifo")
curriculum = parsed_config.get("curriculum")
strategy = parsed_config.get("strategy")
train_seed = parsed_config.get("train_seed")
checkpoint_step = parsed_config.get("checkpoint_step")
root = Path(parsed_config.get("root") or "/workspace/Options_In_Simulation")

ckpt_path = parsed_config.get("checkpoint")

# If no explicit checkpoint, try to reconstruct from train naming
if ckpt_path is None and all(v is not None for v in (fifo, curriculum, strategy, train_seed, checkpoint_step)):
    run_base = f"{algo}_{action_dim}_{fifo}_{curriculum}_{strategy}"
    leaf = f"{run_base}_s{train_seed}"
    ckpt_path = root / "my_runs" / run_base / leaf / "checkpoints" / f"agent_{checkpoint_step}.pt"
    ckpt_path = str(ckpt_path.resolve())

# Extract step if still unknown but path given
if checkpoint_step is None and ckpt_path is not None:
    m = re.search(r"agent_(\d+)\.pt$", ckpt_path)
    if m:
        checkpoint_step = int(m.group(1))

# Build eval experiment name if not provided
if parsed_config.get("experiment_name"):
    experiment_name = parsed_config["experiment_name"]
else:
    # default to your preferred verbose, provenance-rich naming
    # eval_kaddpg_1d_{fifo}_{curriculum}_{strategy}_s{train_seed}_a{step}_s{seed}
    fifo_s = fifo or "unknown"
    curriculum_s = curriculum or "unknown"
    strategy_s = strategy or "unknown"
    train_seed_s = train_seed if train_seed is not None else "unknown"
    step_s = checkpoint_step if checkpoint_step is not None else "unknown"
    eval_seed_s = seed
    experiment_name = f"eval_{algo}_{action_dim}_{fifo_s}_{curriculum_s}_{strategy_s}_s{train_seed_s}_a{step_s}_s{eval_seed_s}"

# Wire into cfg + W&B
cfg["experiment"]["experiment_name"] = experiment_name
cfg["experiment"]["wandb_kwargs"]["name"] = experiment_name

# ----------------------------
# Agent & load checkpoint
# ----------------------------
agent = DDPG(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

if ckpt_path:
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    agent.load(ckpt_path)
else:
    print("[WARN] No checkpoint provided or reconstructed; running WITHOUT loading weights.")

# ----------------------------
# Trainer
# ----------------------------
cfg_trainer = {"timesteps": 50_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# ----------------------------
# Start eval
# ----------------------------
trainer.eval()
