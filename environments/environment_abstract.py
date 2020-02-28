from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from random import choice
import torch.nn as nn


class State(ABC):
    pass


class Environment(ABC):
    def __init__(self):
        self.dtype = np.float
        self.fixed_actions: bool = True

    @abstractmethod
    def next_state(self, states: List[State], action: int) -> Tuple[List[State], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param action: Action to take
        @return: Next states, transition costs
        """
        pass

    @abstractmethod
    def prev_state(self, states: List[State], action: int) -> List[State]:
        """ Get the previous state based on the current state and action

        @param states: List of states
        @param action: Action to take to get the previous state
        @return: Previous states
        """
        pass

    @abstractmethod
    def generate_goal_states(self, num_states: int) -> List[State]:
        """ Generate goal states

        @param num_states: Number of states to generate
        @return: List of states
        """
        pass

    @abstractmethod
    def is_solved(self, states: List[State]) -> np.ndarray:
        """ Returns whether or not state is solved

        @param states: List of states
        @return: Boolean numpy array where the element at index i corresponds to whether or not the
        state at index i is solved
        """
        pass

    @abstractmethod
    def state_to_nnet_input(self, states: List[State]) -> List[List[np.ndarray]]:
        """ State to numpy arrays to be fed to the neural network

        @param states: List of states
        @return: List of Lists of numpy arrays. Each state corresponds to a list of numpy arrays
        """
        pass

    @abstractmethod
    def get_moves(self) -> List[int]:
        """ Used for environments with fixed actions. Corresponds to the numbers of each action

        @return: List of action ints
        """
        pass

    @abstractmethod
    def get_nnet_model(self) -> nn.Module:
        """ Get the neural network model for the environment

        @return: neural network model
        """
        pass

    @abstractmethod
    def get_str_rep(self, state: List[State]) -> List[str]:
        """ Get the unique string representation for the state. Used in A* search to uniquely identify states

        @param state: List of states
        @return: List of string representations
        """
        pass

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> List[State]:
        """ Generate training states by starting from the goal and taking actions in reverse.
        If the number of actions are not fixed, then a custom implementation must be used.

        @param num_states: Number of states to generate
        @param backwards_range: Min and max number times to take a move in reverse
        @return: List of states
        """
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        legal_moves: List[int] = self.get_moves()

        # Get goal states
        states: List[State] = self.generate_goal_states(num_states)

        scramble_nums: np.ndarray = np.random.choice(scrambs, num_states)
        num_moves: np.ndarray = np.zeros(num_states)

        # Go backward from goal state
        while np.max(num_moves < scramble_nums):
            idxs: np.ndarray = np.where((num_moves < scramble_nums))[0]
            subset_size: int = int(max(len(idxs) / len(legal_moves), 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = choice(legal_moves)
            states_to_move = [states[i] for i in idxs]
            states_moved = self.prev_state(states_to_move, move)

            for state_moved_idx, state_moved in enumerate(states_moved):
                states[idxs[state_moved_idx]] = state_moved

            num_moves[idxs] = num_moves[idxs] + 1

        return states

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        """ Generate all children for the state

        @param states: List of states
        @return: Children of each state, Transition costs for each state
        """
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        legal_moves: List[int] = self.get_moves()

        states_exp: List[List[State]] = []
        for _ in range(len(states)):
            states_exp.append([])

        tc: np.ndarray = np.empty([num_states, len(legal_moves)])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx, move in enumerate(legal_moves):
            # next state
            states_next_move: List[State]
            tc_move: List[float]
            states_next_move, tc_move = self.next_state(states, move)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(states_next_move[idx])

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l
