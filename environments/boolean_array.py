from typing import List, Tuple, Union
import numpy as np
import torch.nn as nn

from utils.env_utils import create_nnet_with_overridden_params
from .environment_abstract import Environment, State


class BooleanArrayState(State):
    __slots__ = ['booleans', 'hash']

    def __init__(self, booleans: np.ndarray):
        self.booleans: np.ndarray = booleans
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.booleans.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.booleans, other.booleans)


class BooleanArray(Environment):
    def __init__(self, dim: int):
        super().__init__()

        self.dim: int = dim
        if self.dim <= 8:
            self.dtype = np.uint8
        elif self.dim <= 16:
            self.dtype = np.uint16
        else:
            self.dtype = np.uint32

        self.moves: List[str] = [str(i) for i in range(self.dim)]

        self.goal_state = np.zeros(self.dim, dtype=self.dtype)

    def next_state(self, states: List[BooleanArrayState], action: int) -> Tuple[List[BooleanArrayState], List[float]]:
        # initialize
        states_np = np.stack([x.booleans for x in states], axis=0)
        states_next_np: np.ndarray = states_np.copy()

        # get next state
        states_next_np[:, action] = ~states_next_np[:, action]

        # make states
        states_next: List[BooleanArrayState] = [BooleanArrayState(x) for x in list(states_next_np)]
        transition_costs = [1 for _ in range(len(states))]

        return states_next, transition_costs

    def prev_state(self, states: List[BooleanArrayState], action: int) -> List[BooleanArrayState]:
        return self.next_state(states, action)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[BooleanArrayState], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_state.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[BooleanArrayState] = [BooleanArrayState(self.goal_state.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[BooleanArrayState]) -> np.ndarray:
        states_np = np.stack([state.booleans for state in states], axis=0)

        return ~states_np.any(axis=1)

    def state_to_nnet_input(self, states: List[BooleanArrayState]) -> List[np.ndarray]:
        states_np = np.stack([x.booleans for x in states], axis=0)

        representation = [states_np.astype(self.dtype)]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        kwargs = dict(state_dim=self.dim, one_hot_depth=0, h1_dim=5000, resnet_dim=1000,
                      num_resnet_blocks=4, out_dim=1, batch_norm=True)

        return create_nnet_with_overridden_params(kwargs)
