import os
import re
from pathlib import Path

import torch
import torch.nn as nn

# skrl / isaac
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.envs.torch import wrap_env
from skrl.utils import set_seed

# ==== ADJUST IF NEEDED (agent / task modules) ====
# If your KAMMA agent/module path differs, change here:
from my_agents.ddpg import DDPG
from my_trainers.sequential import SequentialTrainer
# ================================================

from argparse import ArgumentParser

# ----------------------------
# (Example) KAMMA models
# ----------------------------
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 128), nn.ELU(),
            nn.Linear(128, self.num_actions), nn.Tanh()  # often wider range for 4D torques
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 128), nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        import torch as _t
        return self.net(_t.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

# ----------------------------
# Env / Task
# ----------------------------
headless = True
env = get_env_instance(headless=headless, enable_livestream=False, enable_viewport=False)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.KAMMA_eval import TorqueDistributionTask, TASK_CFG  # <— adjust module name if different

# ----------------------------
# Args (+ env fallbacks)
# ----------------------------
arg_parser = ArgumentParser(description="Evaluate KAMMA 4D agent")

# physics / control
arg_parser.add_argument("--stiffness", type=float, default=0.035)
arg_parser.add_argument("--damping", type=float, default=0.005)
arg_parser.add_argument("--static_friction", type=float, default=0.85)
arg_parser.add_argument("--dynamic_friction", type=float, default=0.85)
arg_parser.add_argument("--yaw_constant", type=float, default=0.5)
arg_parser.add_argument("--linear_x", type=float, nargs=2, default=[1.0, 2.0])

# eval control
arg_parser.add_argument("--seed", type=int, default=None)
arg_parser.add_argument("--checkpoint", type=str, default=None)
arg_parser.add_argument("--experiment-name", type=str, default=None)

# optional train-reconstruction
arg_parser.add_argument("--train-seed", type=int, default=None)
arg_parser.add_argument("--fifo", choices=["fifo", "nofifo"], default=None)
arg_parser.add_argument("--curriculum", type=str, default=None)
arg_parser.add_argument("--strategy", type=str, default=None)
arg_parser.add_argument("--checkpoint-step", type=int, default=None)
arg_parser.add_argument("--root", type=str, default="/workspace/Options_In_Simulation")

parsed = vars(arg_parser.parse_args())

# env fallbacks
if parsed["seed"] is None and os.getenv("EVAL_SEED"):
    parsed["seed"] = int(os.getenv("EVAL_SEED"))
if parsed["checkpoint"] is None and os.getenv("EVAL_CHECKPOINT"):
    parsed["checkpoint"] = os.getenv("EVAL_CHECKPOINT")
if parsed["experiment_name"] is None and os.getenv("EVAL_EXPERIMENT_NAME"):
    parsed["experiment_name"] = os.getenv("EVAL_EXPERIMENT_NAME")

# ----------------------------
# Seed & TASK config
# ----------------------------
seed = set_seed(parsed["seed"] if parsed["seed"] is not None else 42)

TASK_CFG["seed"] = seed
TASK_CFG["headless"] = headless
TASK_CFG["task"]["env"]["numEnvs"] = 1

# control/friction/commands
TASK_CFG["task"]["env"]["control"]["stiffness"] = parsed["stiffness"]
TASK_CFG["task"]["env"]["control"]["damping"] = parsed["damping"]
TASK_CFG["task"]["sim"]["default_physics_material"]["static_friction"] = parsed["static_friction"]
TASK_CFG["task"]["sim"]["default_physics_material"]["dynamic_friction"] = parsed["dynamic_friction"]
TASK_CFG["task"]["env"]["randomCommandVelocityRanges"]["yaw_constant"] = parsed["yaw_constant"]
TASK_CFG["task"]["env"]["randomCommandVelocityRanges"]["linear_x"] = parsed["linear_x"]

sim_cfg = SimConfig(TASK_CFG)
task = TorqueDistributionTask(name="TorqueDistribution", sim_config=sim_cfg, env=env)
env.set_task(task=task, sim_params=sim_cfg.get_physics_params(), backend="torch", init_sim=True)

env = wrap_env(env, "omniverse-isaacgym")
device = env.device

# ----------------------------
# Memory & Models
# ----------------------------
memory = RandomMemory(memory_size=1_000_000, num_envs=env.num_envs, device=device, replacement=False)
models = {
    "policy": DeterministicActor(env.observation_space, env.action_space, device),
    "target_policy": DeterministicActor(env.observation_space, env.action_space, device),
    "critic": Critic(env.observation_space, env.action_space, device),
    "target_critic": Critic(env.observation_space, env.action_space, device),
}

# ----------------------------
# KAMMA config + logging
# ----------------------------
KAMMA_DEFAULT_CONFIG = {
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
    "exploration": {"noise": False, "initial_scale": 1.0, "final_scale": 1.0, "timesteps": 50_000.0},
    "rewards_shaper": None,
    "mixed_precision": False,
    "experiment": {
        "directory": "/workspace/Options_In_Simulation/my_runs/eval_kamma_4d",
        "experiment_name": None,
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": True,
        "wandb_kwargs": {
            "project": "KAMMA 4D Study",
            "entity": "meesjansen-Delft Technical University",
            "name": None,
            "tags": ["KAMMA", "4D", "o6", "torq"],
            "dir": "/workspace/Options_In_Simulation/my_runs",
        },
    },
}

cfg = KAMMA_DEFAULT_CONFIG.copy()
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
# Checkpoint & experiment-name
# ----------------------------
algo, action_dim = "kamma", "4d"
fifo, curriculum, strategy = parsed.get("fifo"), parsed.get("curriculum"), parsed.get("strategy")
train_seed, step = parsed.get("train_seed"), parsed.get("checkpoint_step")
root = Path(parsed.get("root") or "/workspace/Options_In_Simulation")
ckpt = parsed.get("checkpoint")

if ckpt is None and all(v is not None for v in (fifo, curriculum, strategy, train_seed, step)):
    run = f"{algo}_{action_dim}_{fifo}_{curriculum}_{strategy}"
    leaf = f"{run}_s{train_seed}"
    ckpt = str((root / "my_runs" / run / leaf / "checkpoints" / f"agent_{step}.pt").resolve())

if step is None and ckpt is not None:
    m = re.search(r"agent_(\d+)\.pt$", ckpt)
    if m: step = int(m.group(1))

if parsed.get("experiment_name"):
    exp_name = parsed["experiment_name"]
else:
    exp_name = f"eval_{algo}_{action_dim}_{fifo or 'unknown'}_{curriculum or 'unknown'}_{strategy or 'unknown'}_s{train_seed if train_seed is not None else 'unknown'}_a{step if step is not None else 'unknown'}_s{seed}"

cfg["experiment"]["experiment_name"] = exp_name
cfg["experiment"]["wandb_kwargs"]["name"] = exp_name

# ----------------------------
# Agent + load
# ----------------------------
agent = DDPG(models=models, memory=memory, cfg=cfg,
              observation_space=env.observation_space, action_space=env.action_space, device=device)

if ckpt:
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"[INFO] Loading checkpoint: {ckpt}")
    agent.load(ckpt)
else:
    print("[WARN] No checkpoint provided/reconstructed; evaluating uninitialized weights.")

# ----------------------------
# Trainer
# ----------------------------
cfg_tr = {"timesteps": 50_000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_tr, env=env, agents=agent)
trainer.eval()
