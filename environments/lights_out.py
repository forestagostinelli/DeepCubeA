from typing import List, Tuple, Union
import numpy as np
from torch import nn

from utils.env_utils import create_nnet_with_overridden_params
from .environment_abstract import Environment, State


class LOState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: np.ndarray):
        self.tiles: np.ndarray = tiles
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.tiles.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)


class LightsOut(Environment):
    def __init__(self, dim: int):
        super().__init__()
        self.dtype = np.uint8
        self.dim = dim
        self.num_tiles = self.dim ** 2

        self.move_matrix = np.zeros((self.num_tiles, 5), dtype=np.int64)
        for move in range(self.num_tiles):
            x_pos = int(np.floor(move / self.dim))
            y_pos = move % self.dim

            right = move + self.dim if x_pos < (self.dim-1) else move
            left = move - self.dim if x_pos > 0 else move
            up = move + 1 if y_pos < (self.dim - 1) else move
            down = move - 1 if y_pos > 0 else move

            self.move_matrix[move] = [move, right, left, up, down]

    def next_state(self, states: List[LOState], action: int) -> Tuple[List[LOState], List[float]]:
        states_np = np.stack([x.tiles for x in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, [action] * states_np.shape[0])

        states_next: List[LOState] = [LOState(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[LOState], action: int) -> List[LOState]:
        return self.next_state(states, action)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[LOState], np.ndarray]:
        if np_format:
            solved_states: np.ndarray = np.zeros((num_states, self.num_tiles), dtype=self.dtype)
        else:
            solved_states: List[LOState] = [LOState(np.zeros(self.num_tiles,
                                                             dtype=self.dtype)) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[LOState]) -> np.ndarray:
        states_np = np.stack([state.tiles for state in states], axis=0)

        return np.all(states_np == 0, axis=1)

    def state_to_nnet_input(self, states: List[LOState]) -> List[np.ndarray]:
        states_np = np.stack([state.tiles for state in states], axis=0).astype(self.dtype)

        representation: List[np.ndarray] = [states_np]

        return representation

    def get_num_moves(self) -> int:
        return self.num_tiles

    def get_nnet_model(self) -> nn.Module:
        kwargs = dict(state_dim=self.num_tiles, one_hot_depth=6, h1_dim=5000, resnet_dim=1000,
                      num_resnet_blocks=4, out_dim=1, batch_norm=True)

        return create_nnet_with_overridden_params(kwargs)

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[LOState], List[int]]:
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
        moves = np.random.choice(num_env_moves, size=(num_states, max(scrambs)))
        move_idx: int = 0

        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]

            moves_i: List[int] = list(moves[idxs, move_idx])
            states_np[idxs], _ = self._move_np(states_np[idxs], moves_i)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

            move_idx += 1

        states: List[LOState] = [LOState(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.tiles for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, [move_idx] * states_np.shape[0])

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(LOState(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, actions: List[int]):
        states_next_np: np.ndarray = states_np.copy()

        state_idxs: np.ndarray = np.arange(0, states_next_np.shape[0])
        state_idxs = np.expand_dims(state_idxs, 1)

        move_matrix = self.move_matrix[actions]
        states_next_np[state_idxs, move_matrix] = (states_next_np[state_idxs, move_matrix] + 1) % 2

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs
