import gym
import torch
import torch.nn as nn

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.utils import set_seed

from my_agents.option_critic import *

# Seed for reproducibility
seed = set_seed()  # e.g. `set_seed(42)` for fixed seed

# Define the model
class OptionCriticModel(Model):
    def __init__(self, observation_space, action_space, device, num_options=4, hidden_size=128):
        Model().__init__(self, observation_space, action_space, device)
        
        self.num_options = num_options
        obs_size = self.num_observations
        action_size = self.num_actions  # Assuming discrete actions

        # Feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_size),
            nn.ReLU()
        )

        # Critic for options
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
        q_options = self.Q(features)  # Shape: [batch_size, num_options]

        # Compute termination probabilities
        beta = torch.sigmoid(self.terminations(features))  # Shape: [batch_size, num_options]

        # Compute intra-option policies
        logits_options = []
        for o in range(self.num_options):
            logits = torch.matmul(features, self.options_W[o]) + self.options_b[o]
            pi = torch.softmax(logits, dim=-1)  # Shape: [batch_size, num_actions]
            logits_options.append(pi)
        logits_options = torch.stack(pi_options, dim=1)  # Shape: [batch_size, num_options, num_actions]

        # Option policies as probabilities (softmax over actions)
        pi_options = torch.softmax(logits_options, dim=-1)  # Shape: [batch_size, num_options, num_actions]
        
        return {"q_options": q_options, "beta": beta, "pi_options": pi_options, "logits_options": logits_options}

    def act(self, inputs, role):
        outputs = self.compute(inputs, role)
        q_options = outputs["q_options"][0]  # Shape: [num_options]
        beta = outputs["beta"][0]  # Shape: [num_options]
        pi_options = outputs["pi_options"][0]  # Shape: [num_options, num_actions]

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
        pi = pi_options[option]
        if self.testing:
            action = torch.argmax(pi).unsqueeze(0)
        else:
            action = torch.multinomial(pi, num_samples=1)

        return {"actions": action, "option": torch.tensor([option])}

    def reset(self):
        self.current_option = None
        self.current_option_terminated = True


        
# instance VecEnvBase and setup task
headless = True  # set headless to False for rendering
env = get_env_instance(headless=headless, enable_livestream=True, enable_viewport=True)

sim_config = SimConfig(TASK_CFG)
task = ReachingTargetTask(name="ReachingTarget", sim_config=sim_config, env=env)
env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=True)

from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from my_envs.DQN_terrain import ReachingTargetTask, TASK_CFG
# wrap the environment
env = wrap_env(env, "omniverse-isaacgym")

device = env.device

# Create the memory
memory = RandomMemory(memory_size=10000, num_envs=1, device=device)

# Instantiate the model
num_options = 5
model = OptionCriticModel(env.observation_space, env.action_space, num_options).to(device)
target_model = OptionCriticModel(env.observation_space, env.action_space, num_options).to(device)
target_model.load_state_dict(model.state_dict())

models = {
    "model": model,
    "target_model": target_model
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
    "gradient_clipping": 1.0
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




