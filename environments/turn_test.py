# import environments, and cube3
import numpy as np
from typing import List, Dict, Tuple, Union
from environments.cube3 import Cube3

cube_env = Cube3()

# Generate solved state
a = cube_env.generate_goal_states(1)
# a[0] is the first cube state, get the cube colors
cube_np = a[0].colors
print("start", cube_np)

# Wrap in list to make it a list of cube colors (what the environment expects)
cube_state_list = np.array([cube_np])
# print(cube_state_list)

next, _ = cube_env._move_np(cube_state_list, 0) # U
next, _ = cube_env._move_np(next, 1) # U'

# Print the cube state
print("end", next[0])



