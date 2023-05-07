from typing import Tuple, List

from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from environments.environment_abstract import Environment, State

from utils.pytorch_models import ResnetModel

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


def load_states(file_name: str) -> List[SokobanState]:
    states_np = pickle.load(open(file_name, "rb"))
    states: List[SokobanState] = []

    agent_idxs = np.where(states_np == 1)
    box_masks = states_np == 2
    goal_masks = states_np == 3
    wall_masks = states_np == 4

    for idx in range(states_np.shape[0]):
        agent_idx = np.array([agent_idxs[1][idx], agent_idxs[2][idx]], dtype=np.int)

        states.append(SokobanState(agent_idx, box_masks[idx], wall_masks[idx], goal_masks[idx]))

    return states


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

        # self.states_train: List[SokobanState] = load_states("data/sokoban/train/data_0.pkl")

    @property
    def num_actions_max(self):
        return self.num_moves

    def get_num_moves(self) -> int:
        return self.num_moves

    def rand_action(self, states: List[State]) -> List[int]:
        return list(np.random.randint(0, self.num_moves, size=len(states)))

    def next_state(self, states: List[SokobanState], actions: List[int]) -> Tuple[List[SokobanState], List[float]]:
        agent = np.stack([state.agent for state in states], axis=0)
        boxes = np.stack([state.boxes for state in states], axis=0)
        walls_next = np.stack([state.walls for state in states], axis=0)

        idxs_arange = np.arange(0, len(states))
        agent_next_tmp = self._get_next_idx(agent, actions)
        agent_next = np.zeros(agent_next_tmp.shape, dtype=np.int)

        boxes_next = boxes.copy()

        # agent -> wall
        agent_wall = walls_next[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        agent_next[agent_wall] = agent[agent_wall]

        # agent -> box
        agent_box = boxes[idxs_arange, agent_next_tmp[:, 0], agent_next_tmp[:, 1]]
        boxes_next_tmp = self._get_next_idx(agent_next_tmp, actions)

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
        agent_mat: np.ndarray = np.zeros((len(states), self.dim, self.dim), dtype=np.int)
        for idx, state in enumerate(states):
            agent_mat[idx, state.agent[0], state.agent[1]] = 1

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

        states_train: List[SokobanState] = load_states("data/sokoban/train/data_0.pkl")
        state_idxs = np.random.randint(0, len(states_train), size=num_states)
        states_seed: List[SokobanState] = [states_train[idx] for idx in state_idxs]
        states, _ = self._random_walk(states_seed, (0, 100))
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

        # Go backward from goal state
        steps_lt = step_nums_curr < step_nums
        while np.any(steps_lt):
            idxs: np.ndarray = np.where(steps_lt)[0]

            states_to_move: List[SokobanState] = [states[idx] for idx in idxs]
            actions = list(np.random.randint(0, self.num_moves, size=len(states_to_move)))

            states_moved, _ = self.next_state(states_to_move, actions)

            for idx_moved, idx in enumerate(idxs):
                states[idx] = states_moved[idx_moved]

            step_nums_curr[idxs] = step_nums_curr[idxs] + 1
            steps_lt[idxs] = step_nums_curr[idxs] < step_nums[idxs]

        return states, list(step_nums)

    def _get_next_idx(self, curr_idxs: np.ndarray, actions: List[int]) -> np.ndarray:
        actions_np: np.array = np.array(actions)
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

            self.state = self.env.next_state([self.state], [action])[0][0]
            self._update_plot()
            if self.env.is_solved([self.state])[0]:
                print("SOLVED!")
        elif event.key.upper() in 'R':
            self._get_instance()
            self._update_plot()
        elif event.key.upper() in 'P':
            for i in range(1000):
                action = self.env.rand_action([self.state])[0]
                self.state = self.env.next_state([self.state], [action])[0][0]
            self._update_plot()


def main():
    env: Sokoban = Sokoban(10, 4)

    fig = plt.figure(figsize=(5, 5))
    interactive_env = InteractiveEnv(env, fig)
    fig.add_axes(interactive_env)

    plt.show()


if __name__ == '__main__':
    main()
