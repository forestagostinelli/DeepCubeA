from typing import List, Tuple
import numpy as np
import torch.nn as nn

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State


class NPuzzleState(State):
    __slots__ = ['tiles']

    def __init__(self, tiles: np.ndarray):
        self.tiles: np.ndarray = tiles
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.tiles.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)


class NPuzzle(Environment):
    moves: List[str] = ['U', 'D', 'L', 'R']
    moves_rev: List[str] = ['D', 'U', 'R', 'L']

    def __init__(self, dim: int):
        super().__init__()

        self.dim: int = dim
        if self.dim <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int

        # Solved state
        self.goal_tiles: np.ndarray = np.concatenate((np.arange(1, self.dim * self.dim), [0])).astype(self.dtype)

        # Next state ops
        self.swap_zero_idxs: np.ndarray = self._get_swap_zero_idxs(self.dim)

    def next_state(self, states: List[NPuzzleState], action: int) -> Tuple[List[NPuzzleState], List[float]]:
        # initialize
        states_np = np.stack([x.tiles for x in states], axis=0)
        states_next_np: np.ndarray = states_np.copy()

        # get zero indicies
        z_idxs: np.ndarray
        _, z_idxs = np.where(states_next_np == 0)

        # get index to swap with zero
        state_idxs: np.ndarray = np.arange(0, states_next_np.shape[0])
        swap_z_idxs: np.ndarray = self.swap_zero_idxs[z_idxs, action]

        # swap zero with adjacent time
        states_next_np[state_idxs, z_idxs] = states_np[state_idxs, swap_z_idxs]
        states_next_np[state_idxs, swap_z_idxs] = 0

        states_next: List[NPuzzleState] = [NPuzzleState(x) for x in list(states_next_np)]

        transition_costs: List[float] = [1.0 for _ in states]

        return states_next, transition_costs

    def prev_state(self, states: List[NPuzzleState], action: int) -> List[NPuzzleState]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int) -> List[NPuzzleState]:
        solved_states: List[NPuzzleState] = [NPuzzleState(self.goal_tiles.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[NPuzzleState]) -> np.ndarray:
        states_np = np.stack([state.tiles for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_tiles, 0))

        return np.all(is_equal, axis=1)

    def state_to_nnet_input(self, states: List[NPuzzleState]) -> List[np.ndarray]:
        states_np = np.stack([x.tiles for x in states], axis=0)

        representation = [states_np.astype(self.dtype)]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        state_dim: int = self.dim * self.dim
        nnet = ResnetModel(state_dim, self.dim ** 2, 5000, 1000, 4, 1)

        return nnet

    @staticmethod
    def _get_swap_zero_idxs(n: int) -> np.ndarray:
        swap_zero_idxs: np.ndarray = np.zeros((n ** 2, len(NPuzzle.moves)), dtype=np.int)
        for moveIdx, move in enumerate(NPuzzle.moves):
            for i in range(n):
                for j in range(n):
                    z_idx = np.ravel_multi_index((i, j), (n, n))

                    state = np.ones((n, n), dtype=np.int)
                    state[i, j] = 0

                    is_eligible: bool = False
                    if move == 'U':
                        is_eligible = i < (n - 1)
                    elif move == 'D':
                        is_eligible = i > 0
                    elif move == 'L':
                        is_eligible = j < (n - 1)
                    elif move == 'R':
                        is_eligible = j > 0

                    if is_eligible:
                        swap_i: int = -1
                        swap_j: int = -1
                        if move == 'U':
                            swap_i = i + 1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i - 1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j + 1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j - 1

                        swap_zero_idxs[z_idx, moveIdx] = np.ravel_multi_index((swap_i, swap_j), (n, n))
                    else:
                        swap_zero_idxs[z_idx, moveIdx] = z_idx

        return swap_zero_idxs
