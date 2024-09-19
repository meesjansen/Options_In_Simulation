import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your custom environment
from Rooms_Env import ReachingFoodTask

import hypothesis
import hypothesis.strategies as st
import pytest
import warnings
import torch
from skrl.envs.torch import Wrapper, wrap_env

@pytest.fixture
def classes_and_kwargs():
    return []

@pytest.mark.parametrize("wrapper", ["gym", "gymnasium", "dm", "robosuite", \
           "isaacgym-preview2", "isaacgym-preview3", "isaacgym-preview4", "omniverse-isaacgym"])
def test_wrap_env(capsys, classes_and_kwargs, wrapper):
    env = ReachingFoodTask(num_envs=1)

    try:
        env: Wrapper = wrap_env(env=env, wrapper=wrapper)
    except ValueError as e:
        warnings.warn(f"{e}. This test will be skipped for '{wrapper}'")
    except ModuleNotFoundError as e:
        warnings.warn(f"{e}. The '{wrapper}' wrapper module is not found. This test will be skipped")

    # Test environment properties
    env.observation_space
    env.action_space
    env.state_space
    env.num_envs
    env.device
