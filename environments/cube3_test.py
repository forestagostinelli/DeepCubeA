# import environments, and cube3
import numpy as np

from cube3 import Cube3

cube_env = Cube3()

a = cube_env.generate_goal_states(2)
a[0].colors[0] = 1

print(a[0].colors)
print(cube_env.is_solved(a))