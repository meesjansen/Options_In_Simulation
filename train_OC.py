import gym
import torch
import torch.nn as nn


from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.utils import set_seed

from my_models.deterministic import Shape
from my_agents.option_critic import *

# Seed for reproducibility
seed = set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# Define the model
class OptionCriticModel(Model):
    def __init__(self, observation_space, action_space, device, num_options=4, hidden_size=128):
        Model().__init__(self, observation_space, action_space, device)
        
        self.num_options = num_options
        obs_size = self.num_observations
        action_size = self.num_actions  # Assuming discrete actions

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_size),
            nn.ReLU()
        )

        # Critic for options DQN
        self.q_layer = nn.Linear(hidden_size, num_options)
        
        # Termination functions
        self.termination_layer = nn.Linear(hidden_size, num_options)
        
        # Intrta-option policies
        self.options_W = nn.Parameter(torch.zeros(num_options, hidden_size, action_size))
        self.options_b = nn.Parameter(torch.zeros(num_options, action_size))

        
    def compute(self, inputs, role):
        states = inputs["states"]  # Shape: [batch_size, num_observations]
        features = self.feature_extractor(states)  # Shape: [batch_size, 64]

        # Compute option-values
        so_values = self.q_layer(features)  # Shape: [batch_size, num_options]

        # Compute termination probabilities
        beta_termin = torch.sigmoid(self.termination_layer(features))  # Shape: [batch_size, num_options]

        # Compute intra-option policies
        logits_intra = []
        pi_intra = []
        log_probs_intra = []
        for o in range(self.num_options):
            logits = torch.matmul(features, self.options_W[o]) + self.options_b[o]
            logits_intra.append(logits)  # Collect logits

            pi = torch.softmax(logits, dim=-1)  # Shape: [batch_size, num_actions]
            pi_intra.append(pi)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs_intra.append(log_probs)

        # Stack logits and probabilities into tensors
        logits_intra = torch.stack(logits_intra, dim=1)  # Shape: [batch_size, num_options, num_actions]
        pi_intra = torch.stack(pi_intra, dim=1)          # Shape: [batch_size, num_options, num_actions]
        log_probs_intra = torch.stack(log_probs_intra, dim=1)


        return {"q_options": so_values, "beta": beta_termin, "intra_option_pi": pi_intra}

    def act(self, inputs, role):
        
        
        
        outputs = self.compute(inputs, role)
        q_options = outputs["q_options"][0]  # Shape: [num_options]
        beta = outputs["beta"][0]  # Shape: [num_options]
        pi_intra = outputs["intra_option_pi"][0]  # Shape: [num_options, num_actions]

        # Option selection
        if self.current_option is None or self.current_option_terminated:
            pi_o = torch.softmax(q_options / self.temperature, dim=-1)
            if self.testing:
                option = torch.argmax(pi_o).item()
            else:
                option = torch.multinomial(pi_o, num_samples=1).item()
            self.current_option = option
        else:
            option = self.current_option

        # Check for option termination
        termination_prob = beta[option].item()
        if torch.rand(1).item() < termination_prob:
            self.current_option_terminated = True
        else:
            self.current_option_terminated = False

        # Intra-option policy
        actions = pi_intra[option]
        

        return actions # log prob? check act gaussian mix and deterministic mixin

    def reset(self):
        self.current_option = None
        self.current_option_terminated = True


    
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

# Create the memory
memory = RandomMemory(memory_size=300_000, num_envs=env.num_envs, device=device, replacement=False)

# Instantiate the model
num_options = 5
model = OptionCriticModel(env.observation_space, env.action_space, num_options).to(device)
prime_model = OptionCriticModel(env.observation_space, env.action_space, num_options).to(device)
prime_model.load_state_dict(model.state_dict())

models = {
    "Option-Critic": model,
    "Prime Option-Critic": prime_model
}

# Agent configuration
cfg_agent = {
    "learning_rate": 1e-3,
    "discount_factor": 0.99,
    "entropy_coefficient": 0.01,
    "batch_size": 64,
    "num_options": num_options,
    "target_update_frequency": 1000,
    "start_learning": 1000,
    "gradient_clipping": 1.0,
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

# Create the agent
agent = OptionCriticAgent(models=models, memory=memory, cfg=cfg_agent)

# Configure and instantiate the trainer
cfg_trainer = {
    "timesteps": 20000,
    "headless": True
}

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# Start training
trainer.train()




