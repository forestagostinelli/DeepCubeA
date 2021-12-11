from typing import List, Dict, Tuple, Union
import numpy as np
from torch import nn

from utils.env_utils import create_nnet_with_overridden_params
from environments.environment_abstract import Environment, State

from environments.py222 import initState, doAlgStr, isSolved


class Cube2State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube2(Environment):
    moves: List[str] = ["%s%s" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in ['', '\'', '2']]
    moves_rev: List[str] = ["%s%s" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in ['\'', '', '2']]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 2

        # solved state
        self.goal_colors: np.ndarray = initState()

    def next_state(self, states: List[Cube2State], action: int) -> Tuple[List[Cube2State], List[float]]:
        # print([x.colors.shape for x in states])
        # print([x.colors for x in states])
        states_np = np.stack([x.colors for x in states], axis=0)
        states_next_np = self._move_np(states_np, action)

        states_next: List[Cube2State] = [Cube2State(states_next_np[0][i]) for i in range(states_next_np[0].shape[0])]
        transition_costs = [1 for _ in range(len(states))]

        return states_next, transition_costs

    def prev_state(self, states: List[Cube2State], action: int) -> List[Cube2State]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube2State], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube2State] = [Cube2State(self.goal_colors.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[Cube2State]) -> np.ndarray:
        # states_np = np.stack([state.colors for state in states], axis=0)
        # is_equal = np.equal(states_np, np.expand_dims(self.goal_colors, 0))
        #
        # return np.all(is_equal, axis=1)

        # TODO: Vectorize this.
        is_solved = np.empty(len(states), dtype=self.dtype)
        for i in range(len(is_solved)):
            is_solved[i] = isSolved(states[i].colors)
        return is_solved

    def state_to_nnet_input(self, states: List[Cube2State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)

        # TODO: Figure out if this should be done for cube2 (it was done for cube3).
        representation_np: np.ndarray = states_np / (self.cube_len ** 2)
        representation_np: np.ndarray = representation_np.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        kwargs = dict(state_dim=(self.cube_len ** 2) * 6, one_hot_depth=6, h1_dim=5000, resnet_dim=1000,
                      num_resnet_blocks=4, out_dim=1, batch_norm=True)

        return create_nnet_with_overridden_params(kwargs)

    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]

        # TODO: Vectorize this.
        states_next_np: np.ndarray = states_np.copy()
        for i in range(states_next_np.shape[0]):
            states_next_np[i] = doAlgStr(states_next_np[i], action_str)

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

if __name__ == "__main__":
    env = Cube2()
    goal_states = env.generate_goal_states(10)
    next_states = env.next_state(goal_states, 0)
    next_states2 = env.next_state(next_states[0], 0)
    is_solved = env.is_solved(next_states2[0])
    print(env.moves)
    print(env.moves_rev)
    print('Goal states:')
    print(goal_states[0].colors)
    print(goal_states[1].colors)
    print(goal_states[2].colors)
    print('Next states:')
    print(next_states[0][0].colors)
    print(next_states[0][1].colors)
    print(next_states[0][2].colors)
    print('Next states 2:')
    print(next_states2[0][0].colors)
    print(next_states2[0][1].colors)
    print(next_states2[0][2].colors)
    print('Is solved:')
    print(is_solved[0])
    print(is_solved[1])
    print(is_solved[2])

