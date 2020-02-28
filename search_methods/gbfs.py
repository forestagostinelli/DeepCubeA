from typing import List, Tuple, Set, Callable, Optional
from environments.environment_abstract import Environment, State
import numpy as np
import time
from utils import search_utils, env_utils, nnet_utils
import glob
import pickle
from argparse import ArgumentParser

import torch


class Instance:

    def __init__(self, state: State, eps: float):
        self.curr_state: State = state
        self.is_solved: bool = False
        self.num_steps: int = 0
        self.trajs: List[Tuple[State, float]] = []
        self.seen_states: Set[str] = set()

        self.eps = eps

    def add_to_traj(self, state: State, cost_to_go: float, str_rep: str):
        self.trajs.append((state, cost_to_go))
        self.seen_states.add(str_rep)

    def next_state(self, state: State):
        self.curr_state = state
        self.num_steps += 1

    def seen_state(self, str_rep: str):
        return str_rep in self.seen_states


class GBFS:
    def __init__(self, states: List[State], env: Environment, eps: Optional[List[float]] = None):
        self.curr_states: List[State] = states
        self.env: Environment = env

        if eps is None:
            eps = [0] * len(self.curr_states)

        self.instances: List[Instance] = []
        for state, eps_inst in zip(states, eps):
            instance: Instance = Instance(state, eps_inst)
            self.instances.append(instance)

    def step(self, heuristic_fn: Callable) -> None:
        # check which are solved
        self._record_solved()

        # take a step for unsolved states
        self._move(heuristic_fn)

    def get_trajs(self) -> List[List[Tuple[State, float]]]:
        trajs_all: List[List[Tuple[State, float]]] = []
        for instance in self.instances:
            trajs_all.append(instance.trajs)

        return trajs_all

    def get_is_solved(self) -> List[bool]:
        is_solved: List[bool] = [x.is_solved for x in self.instances]

        return is_solved

    def get_num_steps(self) -> List[int]:
        num_steps: List[int] = [x.num_steps for x in self.instances]

        return num_steps

    def _record_solved(self) -> None:
        # get unsolved instances
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return

        states: List[State] = [instance.curr_state for instance in instances]

        is_solved: np.ndarray = self.env.is_solved(states)

        solved_idxs: List[int] = list(np.where(is_solved)[0])
        if len(solved_idxs) > 0:
            instances_solved: List[Instance] = [instances[idx] for idx in solved_idxs]
            states_solved: List[State] = [instance.curr_state for instance in instances_solved]
            str_reps: List[str] = self.env.get_str_rep(states_solved)

            for instance, state, str_rep in zip(instances_solved, states_solved, str_reps):
                instance.add_to_traj(state, 0.0, str_rep)
                instance.is_solved = True

    def _move(self, heuristic_fn: Callable) -> None:
        # get unsolved instances
        instances: List[Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return
        states: List[State] = [instance.curr_state for instance in instances]

        # get backed-up ctg and cost of each move
        ctg_backups: np.ndarray
        ctg_next_p_tcs: List[np.ndarray]
        states_exp: List[List[State]]
        ctg_backups, ctg_next_p_tcs, state_exp = search_utils.bellman(states, heuristic_fn, self.env)

        # make move
        str_reps: List[str] = self.env.get_str_rep(states)
        for idx in range(len(instances)):
            # add state to trajectory
            instance: Instance = instances[idx]
            state: State = states[idx]
            ctg_backup: float = ctg_backups[idx]
            str_rep: str = str_reps[idx]

            instance.add_to_traj(state, ctg_backup, str_rep)

            # get next state
            ctg_next_p_tc: np.ndarray = ctg_next_p_tcs[idx]
            states_next: List[State] = state_exp[idx]
            state_next: State = states_next[int(np.argmin(ctg_next_p_tc))]

            # if next state in trajectory or epsilon random, make random move
            # str_rep_next: str = self.env.get_str_rep([state_next])[0]
            # seen_state: bool = instance.seen_state(str_rep_next)
            eps_rand_move = np.random.random(1)[0] < instance.eps
            if eps_rand_move:
                rand_state_idx = np.random.choice(len(states_next))
                state_next = states_next[rand_state_idx]

            instance.next_state(state_next)

    def _get_unsolved_instances(self) -> List[Instance]:
        instances_unsolved: List[Instance] = [instance for instance in self.instances if not instance.is_solved]
        return instances_unsolved


def gbfs_test(data_dir: str, env: Environment, heuristic_fn: Callable, max_solve_steps: Optional[int] = None):
    data_files: List[str] = glob.glob("%s/*.pkl" % data_dir)

    # load data
    states: List[State] = []
    num_back_steps: List[int] = []
    for data_file in data_files:
        data = pickle.load(open(data_file, "rb"))

        states.extend(data['states'])
        num_back_steps.extend(data['num_back_steps'])

    if max_solve_steps is None:
        max_solve_steps = max(max(num_back_steps), 1)

    # Do GBFS for each back step
    print("Solving states with GBFS with %i steps" % max_solve_steps)
    for back_step_test in range(max(num_back_steps) + 1):
        solve_start_time = time.time()

        # Generate environments
        states_back_step: List[State] = [state for state, back_step in zip(states, num_back_steps) if
                                         back_step == back_step_test]

        # Get state cost-to-go
        state_ctg: np.ndarray = heuristic_fn(states_back_step)

        # Solve with GBFS
        gbfs = GBFS(states_back_step, env, eps=None)
        for _ in range(max_solve_steps):
            gbfs.step(heuristic_fn)

        is_solved: np.ndarray = np.array(gbfs.get_is_solved())
        num_steps: np.ndarray = np.array(gbfs.get_num_steps())

        # Get stats
        per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
        avg_solve_steps = 0.0
        if per_solved > 0.0:
            avg_solve_steps = np.mean(num_steps[is_solved])

        solve_elapsed_time = time.time() - solve_start_time

        # Print results
        print("Back Steps: %i, %%Solved: %.2f, avgSolveSteps: %.2f, StateCTG Mean(Std/Min/Max): %.2f("
              "%.2f/%.2f/%.2f), time: %.2f" % (
                  back_step_test, per_solved, avg_solve_steps, float(np.mean(state_ctg)),
                  float(np.std(state_ctg)), np.min(state_ctg),
                  np.max(state_ctg), solve_elapsed_time))


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of data")

    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--max_steps', type=int, default=None, help="Maximum number ofsteps to take when solving "
                                                                    "with GBFS. If none is given, then this "
                                                                    "is set to the maximum number of "
                                                                    "backwards steps taken to create the "
                                                                    "data")

    args = parser.parse_args()

    # environment
    env: Environment = env_utils.get_environment(args.env)

    # get device and nnet
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=False)

    gbfs_test(args.data_dir, env, heuristic_fn, max_solve_steps=args.max_steps)


if __name__ == "__main__":
    main()
