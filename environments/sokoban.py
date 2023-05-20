from typing import Tuple, List, Optional

from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from environments.environment_abstract import Environment, State

from utils.pytorch_models import ResnetModel
from random import randrange

import pickle


class SokobanState(State):
    __slots__ = ['agent', 'walls', 'boxes', 'goals', 'hash']

    def __init__(self, agent: np.array, boxes: np.ndarray, walls: np.ndarray, goals: np.ndarray):
        self.agent: np.array = agent
        self.walls: np.ndarray = walls
        self.boxes: np.ndarray = boxes
        self.goals: np.ndarray = goals

        self.hash = None

    def __hash__(self):
        if self.hash is None:
            walls_flat = self.walls.flatten()
            boxes_flat = self.boxes.flatten()
            goals_flat = self.goals.flatten()
            state = np.concatenate((self.agent, walls_flat, boxes_flat, goals_flat), axis=0)

            self.hash = hash(state.tobytes())

        return self.hash

    def __eq__(self, other):
        agents_eq: bool = np.array_equal(self.agent, other.agent)
        walls_eq: bool = np.array_equal(self.walls, other.walls)
        boxes_eq: bool = np.array_equal(self.boxes, other.boxes)
        goals_eq: bool = np.array_equal(self.goals, other.goals)

        return agents_eq and walls_eq and boxes_eq and goals_eq


class Sokoban(Environment):

    def generate_goal_states(self, num_states: int) -> List[State]:
        # not necessary to implement
        raise NotImplementedError

    def prev_state(self, states: List[State], action: int) -> List[State]:
        # not necessary to implement
        raise NotImplementedError

    def __init__(self, dim: int, num_boxes: int):
        super().__init__()

        self.dim: int = dim
        self.num_boxes: int = num_boxes

        self.num_moves: int = 4

        states_train_np: np.ndarray = pickle.load(open("data/sokoban/train/data_0.pkl", "rb"))
        self.agent_train_idxs = np.where(states_train_np == 1)
        self.box_train_masks = states_train_np == 2
        self.goal_train_masks = states_train_np == 3
        self.wall_train_masks = states_train_np == 4
        self.states_train: Optional[List[SokobanState]] = None

    def get_num_moves(self) -> int:
        return self.num_moves

    def rand_action(self, states: List[State]) -> List[int]:
        return list(np.random.randint(0, self.num_moves, size=len(states)))

    def next_state(self, states: List[SokobanState], action: int) -> Tuple[List[SokobanState], List[float]]:
        agent = np.stack([state.agent for state in states], axis=0)
        boxes = np.stack([state.boxes for state in states], axis=0)
        walls_next = np.stack([state.walls for state in states], axis=0)

        idxs_arange = np.arange(0, len(states))
        agent_next_tmp = self._get_next_idx(agent, action)
        agent_next = np.zeros(agent_next_tmp.shape, dtype=np.int)

        boxes_next = boxes.copy()

        # agent -> wall
        agent_wall = walls_next[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        agent_next[agent_wall] = agent[agent_wall]

        # agent -> box
        agent_box = boxes[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        boxes_next_tmp = self._get_next_idx(agent_next_tmp, action)

        box_wall = walls_next[idxs_arange, boxes_next_tmp[:, 0], boxes_next_tmp[:, 1]]
        box_box = boxes[idxs_arange, boxes_next_tmp[:, 0], boxes_next_tmp[:, 1]]

        # agent -> box -> obstacle
        agent_box_obstacle = agent_box & (box_wall | box_box)
        agent_next[agent_box_obstacle] = agent[agent_box_obstacle]

        # agent -> box -> empty
        agent_box_empty = agent_box & np.logical_not(box_wall | box_box)
        agent_next[agent_box_empty] = agent_next_tmp[agent_box_empty]
        abe_idxs = np.where(agent_box_empty)[0]

        agent_next_idxs_abe = agent_next[agent_box_empty]
        boxes_next_idxs_abe = boxes_next_tmp[agent_box_empty]

        boxes_next[abe_idxs, agent_next_idxs_abe[:, 0], agent_next_idxs_abe[:, 1]] = False
        boxes_next[abe_idxs, boxes_next_idxs_abe[:, 0], boxes_next_idxs_abe[:, 1]] = True

        # agent -> empty
        agent_empty = np.logical_not(agent_wall | agent_box)
        agent_next[agent_empty] = agent_next_tmp[agent_empty]
        boxes_next[agent_empty] = boxes[agent_empty]

        states_next: List[SokobanState] = []
        for idx, state in enumerate(states):
            state_next: SokobanState = SokobanState(agent_next[idx], boxes_next[idx], walls_next[idx],
                                                    state.goals.copy())
            states_next.append(state_next)

        transition_costs: List[int] = [1 for _ in range(len(states))]

        return states_next, transition_costs

    def state_to_nnet_input(self, states: List[SokobanState]):
        agent_mat: np.ndarray = np.zeros((len(states), self.dim, self.dim), dtype=np.bool)
        for idx, state in enumerate(states):
            agent_mat[idx, state.agent[0], state.agent[1]] = True

        walls_mat = np.stack([x.walls for x in states], axis=0)
        boxes_mat = np.stack([x.boxes for x in states], axis=0)
        goals_mat = np.stack([x.goals for x in states], axis=0)

        states_np = np.stack((agent_mat, walls_mat, boxes_mat, goals_mat), axis=1)
        states_np = states_np.reshape((len(states), -1))

        return [states_np]

    def get_nnet_model(self) -> nn.Module:
        nnet = ResnetModel(4 * 10 * 10, 0, 5000, 1000, 4, 1, True)

        return nnet

    def is_solved(self, states: List[SokobanState]) -> np.array:
        boxes = np.stack([state.boxes for state in states], axis=0)
        goals = np.stack([state.goals for state in states], axis=0)

        return np.all(boxes == goals, axis=(1, 2))

    def get_render_array(self, state: SokobanState) -> np.ndarray:
        state_rendered = np.ones((self.dim, self.dim), dtype=np.int)
        state_rendered -= state.walls
        state_rendered[state.agent[0], state.agent[1]] = 2
        state_rendered += state.boxes * 2
        state_rendered += state.goals * 3

        return state_rendered

    def generate_states(self, num_states: int, step_range: Tuple[int, int]) -> Tuple[List[SokobanState], List[int]]:
        assert (num_states > 0)
        assert (step_range[0] >= 0)

        if self.states_train is None:
            self.states_train = self._get_processed_states()

        state_idxs = np.random.randint(0, len(self.states_train), size=num_states)
        states_seed: List[SokobanState] = [self.states_train[idx] for idx in state_idxs]

        states, _ = self._random_walk(states_seed, (1, 100))
        states_goal, num_steps_l = self._random_walk(states, step_range)

        goals_mat = np.stack([x.boxes for x in states_goal], axis=0)
        for state, goals in zip(states, goals_mat):
            state.goals = goals

        return states, num_steps_l

    def _random_walk(self, states: List[SokobanState],
                     step_range: Tuple[int, int]) -> Tuple[List[SokobanState], List[int]]:
        # Initialize
        num_states: int = len(states)
        scrambs: List[int] = list(range(step_range[0], step_range[1] + 1))
        states = states.copy()

        # Scrambles
        step_nums: np.array = np.random.choice(scrambs, num_states)
        step_nums_curr: np.array = np.zeros(num_states)

        # random walk
        while np.max(step_nums_curr < step_nums):
            idxs: np.ndarray = np.where((step_nums_curr < step_nums))[0]
            subset_size: int = int(max(len(idxs) / self.get_num_moves(), 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(self.get_num_moves())
            states_to_move: List[SokobanState] = [states[i] for i in idxs]
            states_moved: List[SokobanState] = self.next_state(states_to_move, move)[0]

            for state_moved_idx, state_moved in enumerate(states_moved):
                states[idxs[state_moved_idx]] = state_moved

            step_nums_curr[idxs] = step_nums_curr[idxs] + 1

        return states, list(step_nums)

    def _get_next_idx(self, curr_idxs: np.ndarray, action: int) -> np.ndarray:
        actions_np: np.array = np.array([action] * curr_idxs.shape[0])
        next_idxs: np.ndarray = curr_idxs.copy()

        action_idxs = np.where(actions_np == 0)[0]
        next_idxs[action_idxs, 0] = next_idxs[action_idxs, 0] - 1

        action_idxs = np.where(actions_np == 1)[0]
        next_idxs[action_idxs, 0] = next_idxs[action_idxs, 0] + 1

        action_idxs = np.where(actions_np == 2)[0]
        next_idxs[action_idxs, 1] = next_idxs[action_idxs, 1] - 1

        action_idxs = np.where(actions_np == 3)[0]
        next_idxs[action_idxs, 1] = next_idxs[action_idxs, 1] + 1

        next_idxs = np.maximum(next_idxs, 0)
        next_idxs = np.minimum(next_idxs, self.dim-1)

        return next_idxs

    def _get_processed_states(self) -> List[SokobanState]:
        states: List[SokobanState] = []

        for idx in range(self.agent_train_idxs[0].shape[0]):
            agent_idx = np.array([self.agent_train_idxs[1][idx], self.agent_train_idxs[2][idx]], dtype=np.int)

            states.append(SokobanState(agent_idx, self.box_train_masks[idx], self.wall_train_masks[idx],
                                       self.goal_train_masks[idx]))

        return states

    def __getstate__(self):
        self.states_train = None
        return self.__dict__


class InteractiveEnv(plt.Axes):
    def __init__(self, env: Sokoban, fig):
        self.env: Sokoban = env

        super(InteractiveEnv, self).__init__(plt.gcf(), [0, 0, 1, 1])

        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        self.figure.canvas.mpl_connect('key_press_event', self._key_press)

        self._get_instance()
        self._update_plot()

        self.move = []

    def _get_instance(self):
        self.state: SokobanState = self.env.generate_states(1, (1000, 2000))[0][0]

    def _update_plot(self):
        self.clear()
        # rendered_im = self.env.state_to_rgb(self.state)
        # rendered_im_goal = self.env.state_to_rgb(self.state_goal)

        self.imshow(self.env.get_render_array(self.state))
        self.figure.canvas.draw()

    def _key_press(self, event):
        if event.key.upper() in 'ASDW':
            action: int = -1
            if event.key.upper() == 'W':
                action = 0
            if event.key.upper() == 'S':
                action = 1
            if event.key.upper() == 'A':
                action = 2
            if event.key.upper() == 'D':
                action = 3

            # self.states = self.env.next_state(self.states, [action]*len(self.states))[0]
            self.state = self.env.next_state([self.state], action)[0][0]
            self._update_plot()
            if self.env.is_solved([self.state])[0]:
                print("SOLVED!")
        elif event.key.upper() in 'R':
            self._get_instance()
            self._update_plot()
        elif event.key.upper() in 'P':
            for i in range(1000):
                action = self.env.rand_action([self.state])[0]
                self.state = self.env.next_state([self.state], action)[0][0]
            self._update_plot()


def main():
    env: Sokoban = Sokoban(10, 4)

    """
    states, num_steps = env.generate_states(1000, (1, 1))
    states_next, _ = env.expand(states)
    print(sum([max([x != states[idx] for x in states_next[idx]]) for idx in range(len(states))]))
    breakpoint()
    """

    fig = plt.figure(figsize=(5, 5))
    interactive_env = InteractiveEnv(env, fig)
    fig.add_axes(interactive_env)

    plt.show()


if __name__ == '__main__':
    main()
