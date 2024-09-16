import pytest
import torch
from Macro__Agent import MacroQ_LEARNING
from .utils import DummyModel

@pytest.fixture
def agent_kwargs():
    return {"models": {"policy": DummyModel()}}

def test_single_agent(agent_kwargs):
    cfg = {
        "learning_starts": 1,
        "experiment": {"write_interval": 0}
    }
    
    # Instantiate and initialize the agent
    agent = MacroQ_LEARNING(cfg=cfg, **agent_kwargs)

    agent.init()
    agent.pre_interaction(timestep=0, timesteps=1)
    # agent.act(None, timestep=0, timestesps=1)

    # Record dummy transitions
    agent.record_transition(
        states=torch.tensor([]),
        actions=torch.tensor([]),
        rewards=torch.tensor([]),
        next_states=torch.tensor([]),
        terminated=torch.tensor([]),
        truncated=torch.tensor([]),
        infos={},
        timestep=0,
        timesteps=1
    )

    agent.post_interaction(timestep=0, timesteps=1)