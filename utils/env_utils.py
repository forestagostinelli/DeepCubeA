import os
import re
import math

from torch import nn

from environments.environment_abstract import Environment
from utils.pytorch_models import ResnetModel


def get_environment(env_name: str) -> Environment:
    env_name = env_name.lower()
    puzzle_n_regex = re.search("puzzle(\d+)", env_name)
    boolean_array_n_regex = re.search("booleanarray(\d+)", env_name)
    env: Environment

    if env_name == 'cube3':
        from environments.cube3 import Cube3
        env = Cube3()
    elif env_name == 'cube2':
        from environments.cube2 import Cube2
        env = Cube2()
    elif puzzle_n_regex is not None:
        from environments.n_puzzle import NPuzzle
        puzzle_dim: int = int(math.sqrt(int(puzzle_n_regex.group(1)) + 1))
        env = NPuzzle(puzzle_dim)
    elif boolean_array_n_regex is not None:
        from environments.boolean_array import BooleanArray
        puzzle_dim: int = int(boolean_array_n_regex.group(1))
        env = BooleanArray(puzzle_dim)
    elif 'lightsout' in env_name:
        from environments.lights_out import LightsOut
        m = re.search('lightsout([\d]+)', env_name)
        env = LightsOut(int(m.group(1)))
    else:
        raise ValueError('No known environment %s' % env_name)

    return env


def create_nnet_with_overridden_params(kwargs) -> nn.Module:
    param_file = os.environ['NNET_PARAMS']
    print('param_file: ' + param_file)

    import json
    with open(param_file) as f:
        overrides = json.load(f)
        kwargs.update(overrides)
    print('kwargs: ' + str(kwargs))

    nnet = ResnetModel(**kwargs)
    print('nnet: ' + str(nnet))

    return nnet
