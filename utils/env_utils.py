import re
import math
from environments.environment_abstract import Environment


def get_environment(env_name: str) -> Environment:
    env_name = env_name.lower()
    puzzle_n_regex = re.search("puzzle(\d+)", env_name)
    env: Environment

    if env_name == 'cube3':
        from environments.cube3 import Cube3
        env = Cube3()
    elif puzzle_n_regex is not None:
        from environments.n_puzzle import NPuzzle
        puzzle_dim: int = int(math.sqrt(int(puzzle_n_regex.group(1)) + 1))
        env = NPuzzle(puzzle_dim)
    else:
        raise ValueError('No known environment %s' % env_name)

    return env
