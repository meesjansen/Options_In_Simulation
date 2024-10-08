import gym
import torch
import torch.nn as nn

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.omniverse_isaacgym_utils import get_env_instance
from skrl.utils import set_seed
from my_agents.option_critic_v2 import *

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

import gym

env = gym.make("CartPole-v1")

observation_space = env.observation_space
action_space = env.action_space
num_options = 4  # Define the number of options

policy = OptionCriticModel(observation_space, action_space, num_options)

from skrl.memories.torch import ReplayMemory

memory = ReplayMemory(memory_size=10000, 
                      num_envs=1,
                      device=OPTION_CRITIC_DEFAULT_CONFIG["device"])

agent = OptionCriticAgent(models={"policy": policy},
                          memory=memory,
                          observation_space=observation_space,
                          action_space=action_space,
                          device=OPTION_CRITIC_DEFAULT_CONFIG["device"],
                          cfg=OPTION_CRITIC_DEFAULT_CONFIG)

from skrl.trainers.torch import SequentialTrainer

trainer = SequentialTrainer(env=env, agents=agent)
trainer.train()
