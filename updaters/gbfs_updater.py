from typing import List, Tuple, Set
import numpy as np
from utils import nnet_utils, misc_utils
from environments.environment_abstract import Environment, State
from search_methods.gbfs import GBFS
from torch.multiprocessing import Queue, get_context


def gbfs_runner(states_queue: Queue, all_zeros: bool, nnet_dir: str, device, on_gpu: bool, gpu_num: int,
                nnet_batch_size: int, env: Environment, result_queue: Queue, num_steps: int,
                eps_max: float):
    if all_zeros:
        def heuristic_fn(x):
            return np.zeros((len(x)), dtype=np.float)
    else:
        heuristic_fn = nnet_utils.load_heuristic_fn(nnet_dir, device, on_gpu, env.get_nnet_model(), env, clip_zero=True,
                                                    gpu_num=gpu_num, batch_size=nnet_batch_size)

    while True:
        batch_idx: int
        states: List[State]
        batch_idx, states = states_queue.get()
        if batch_idx is None:
            break

        eps: List[float] = list(np.random.rand(len(states)) * eps_max)

        gbfs = GBFS(states, env, eps=eps)
        for _ in range(num_steps):
            gbfs.step(heuristic_fn)

        trajs: List[List[Tuple[State, float]]] = gbfs.get_trajs()

        trajs_flat: List[Tuple[State, float]]
        trajs_flat, _ = misc_utils.flatten_list(trajs)

        is_solved: np.ndarray = np.array(gbfs.get_is_solved())

        states_update: List = []
        cost_to_go_update_l: List[float] = []
        for traj in trajs_flat:
            states_update.append(traj[0])
            cost_to_go_update_l.append(traj[1])

        cost_to_go_update = np.array(cost_to_go_update_l)

        result_queue.put((batch_idx, states_update, cost_to_go_update, is_solved))


class GBFSUpdater:
    def __init__(self, env: Environment, all_zeros: bool, num_procs: int, nnet_dir: str, device, on_gpu: bool,
                 nnet_batch_size: int, num_steps: int, search_batch_size_max: int = 100, eps_max: float = 0.0):
        super().__init__()
        ctx = get_context("spawn")
        self.num_steps = num_steps
        self.search_batch_size_max = search_batch_size_max

        # initialize queues
        self.states_queue: ctx.Queue = ctx.Queue()
        self.result_queue: ctx.Queue = ctx.Queue()

        # initialize processes
        self.num_procs = num_procs
        self.procs: List[ctx.Process] = []
        gpu_nums: List[int] = nnet_utils.get_available_gpu_nums()

        for proc_idx in range(num_procs):
            if len(gpu_nums) > 0:
                gpu_num_idx: int = proc_idx % len(gpu_nums)
                gpu_num = gpu_nums[gpu_num_idx]
            else:
                gpu_num = -1

            proc = ctx.Process(target=gbfs_runner, args=(self.states_queue, all_zeros, nnet_dir, device, on_gpu,
                                                         gpu_num, nnet_batch_size, env, self.result_queue, num_steps,
                                                         eps_max))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def update(self, states: List[State], verbose: bool = False):
        states_update: List[State]
        cost_to_go_update: np.ndarray
        is_solved: np.ndarray
        states_update, cost_to_go_update, is_solved = self._update(states, verbose)

        output_update = np.expand_dims(cost_to_go_update, 1)

        return states_update, output_update, is_solved

    def cleanup(self):
        # join procs
        for _ in self.procs:
            self.states_queue.put((None, None))

        for proc in self.procs:
            proc.join()

    def _update(self, states: List[State], verbose: bool) -> Tuple[List[State], np.ndarray, np.ndarray]:
        # put inputs into runner queue
        num_states: int = len(states)
        num_batches = 0
        start_idx: int = 0
        search_batch_size: int = min(self.search_batch_size_max, int(np.ceil(num_states / self.num_procs)))

        while start_idx < num_states:
            end_idx: int = min(start_idx + search_batch_size, num_states)
            states_batch: List[State] = states[start_idx:end_idx]
            self.states_queue.put((num_batches, states_batch))

            num_batches = num_batches + 1
            start_idx: int = end_idx

        # wait for results
        results: List = [None] * num_batches

        num_batches_finished: int = 0
        progress_points: Set[int] = set(np.linspace(1, num_batches, 10, dtype=np.int))
        for _ in range(num_batches):
            result = self.result_queue.get()
            results[result[0]] = result

            num_batches_finished += 1

            if (num_batches_finished in progress_points) and verbose:
                print("%.2f%%..." % (100.0 * num_batches_finished / num_batches))
        print("")

        # process results
        states_update: List[State] = []
        cost_to_go_update_l: List = []
        is_solved_l: List = []

        for result in results:
            _, states_q, cost_to_go_q, is_solved_q = result
            states_update.extend(states_q)
            cost_to_go_update_l.append(cost_to_go_q)
            is_solved_l.append(is_solved_q)

        cost_to_go_update: np.ndarray = np.concatenate(cost_to_go_update_l, axis=0)
        is_solved: np.ndarray = np.concatenate(is_solved_l, axis=0)

        return states_update, cost_to_go_update, is_solved
