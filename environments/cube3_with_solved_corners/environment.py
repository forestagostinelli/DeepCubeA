
from typing import List, Dict, Tuple, Union
import numpy as np
from torch import nn
from random import randrange

from utils.pytorch_models import ResnetModel
from environments.environment_abstract import Environment, State
from environments.cube3 import Cube3State, Cube3 as Cube3Environment
from xcs229ii_cube.cube2.cube import Cube2
from xcs229ii_cube.cube2.solver import load_lookup_table, find_solution
from xcs229ii_cube.cube3.generated_lists import apply_move_np, MOVES_DEFINITIONS, MOVES_NAMES, MOVES_NAMES_TO_INDICES, REVERSE_MOVES_NAMES, FIXED_CUBIE_MOVES_NAMES_TO_INDICES
from xcs229ii_cube.glue2_to_3 import Glue2To3Cube
from xcs229ii_cube.utils import StickerVectorSerializer

glue = Glue2To3Cube(None)
load_lookup_table()

class Cube3SolvedCorners(Cube3Environment):

    moves: List[str] = [x for x in MOVES_NAMES if x in FIXED_CUBIE_MOVES_NAMES_TO_INDICES]
    moves_rev: List[str] = [x for x in REVERSE_MOVES_NAMES if x in FIXED_CUBIE_MOVES_NAMES_TO_INDICES]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # solved state
        self.goal_colors: np.ndarray = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                                 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=self.dtype)

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Cube3State], List[int]]:
        states_np, scramble_nums = self.generate_3_cube_states_np(num_states, backwards_range)
        oriented_3_cubes = []
        cubes_2_states = glue.convert_3_cubes_to_2_cubes(states_np)
        for i, cube2_state in enumerate(cubes_2_states):
            cube_2_as_int = StickerVectorSerializer(Cube2).unserialize(cube2_state).as_stickers_int
            cube2_solution = find_solution(cube_2_as_int)
            cube3_with_cube2_solution_applied = self.apply_cube2_solution_to_cube3(cube2_solution, states_np[i])
            oriented_3_cubes.append(Cube3State(cube3_with_cube2_solution_applied))

        return oriented_3_cubes, scramble_nums

    def generate_3_cube_states_np(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[np.array, List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        return states_np, scramble_nums.tolist()

    def apply_cube2_solution_to_cube3(self, cube2_solution, cube3_state):
        new_cube3_state = np.array([cube3_state])
        for move_name in cube2_solution:
            move = MOVES_DEFINITIONS[MOVES_NAMES_TO_INDICES[move_name]]
            new_cube3_state = apply_move_np(new_cube3_state, move)
        return new_cube3_state[0]

    def _move_np(self, states_np: np.ndarray, action: int):
        states_next_np = apply_move_np(states_np, MOVES_DEFINITIONS[action])
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Doesn't do anything, it's here just for compat with parent Cube3 class
        """
        return {}, {}

#c = Cube3SolvedCorners()
#print(c.generate_states(20, (2, 2)))
# states = np.stack([state.colors for state in c.generate_goal_states(1)])
# print(c._move_np(states, 1))



