from typing import List, Tuple
import numpy as np
from environments.environment_abstract import Environment, State
from utils import misc_utils


def is_valid_soln(state: State, soln: List[int], env: Environment) -> bool:
    soln_state: State = state
    move: int
    for move in soln:
        soln_state = env.next_state([soln_state], move)[0][0]

    return env.is_solved([soln_state])[0]


def bellman(states: List, heuristic_fn, env: Environment) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    # expand states
    states_exp, tc_l = env.expand(states)
    tc = np.concatenate(tc_l, axis=0)

    # get cost-to-go of expanded states
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    ctg_next: np.ndarray = heuristic_fn(states_exp_flat)

    # backup cost-to-go
    ctg_next_p_tc = tc + ctg_next
    ctg_next_p_tc_l = np.split(ctg_next_p_tc, split_idxs)

    is_solved = env.is_solved(states)
    ctg_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)

    return ctg_backup, ctg_next_p_tc_l, states_exp
