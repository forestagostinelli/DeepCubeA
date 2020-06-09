from typing import List
from argparse import ArgumentParser
import torch
import torch.nn as nn

from environments.environment_abstract import Environment, State
from utils import nnet_utils
from utils import env_utils
from torch.multiprocessing import Queue, get_context

import time


def data_runner(queue1: Queue, queue2: Queue):
    queue1.get()
    queue2.put(1)

    queue1.get()
    queue2.put(1)


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--num_states', type=int, default=100, help="")
    parser.add_argument('--back_max', type=int, default=30, help="")

    args = parser.parse_args()

    # get environment
    env: Environment = env_utils.get_environment(args.env)

    # generate goal states
    start_time = time.time()
    states: List[State] = env.generate_goal_states(args.num_states)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states)/elapsed_time
    print("Generated %i goal states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # get data
    start_time = time.time()
    states: List[State]
    states, _ = env.generate_states(args.num_states, (0, args.back_max))

    elapsed_time = time.time() - start_time
    states_per_sec = len(states)/elapsed_time
    print("Generated %i states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # expand
    start_time = time.time()
    env.expand(states)
    elapsed_time = time.time() - start_time
    states_per_sec = len(states)/elapsed_time
    print("Expanded %i states in %s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # nnet format
    start_time = time.time()

    states_nnet = env.state_to_nnet_input(states)

    elapsed_time = time.time() - start_time
    states_per_sec = len(states)/elapsed_time
    print("Converted %i states to nnet format in "
          "%s seconds (%.2f/second)" % (len(states), elapsed_time, states_per_sec))

    # get heuristic fn
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    nnet: nn.Module = env.get_nnet_model()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet) 
        
    # nnet initialize
    print("")
    heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env)
    heuristic_fn(states)

    # compute
    start_time = time.time()
    heuristic_fn(states)

    nnet_time = time.time() - start_time
    states_per_sec = len(states)/nnet_time
    print("Computed heuristic for %i states in %s seconds (%.2f/second)" % (len(states), nnet_time, states_per_sec))

    # multiprocessing
    print("")
    start_time = time.time()
    ctx = get_context("spawn")
    queue1: ctx.Queue = ctx.Queue()
    queue2: ctx.Queue = ctx.Queue()
    proc = ctx.Process(target=data_runner, args=(queue1, queue2))
    proc.daemon = True
    proc.start()
    print("Process start time: %.2f" % (time.time() - start_time))

    queue1.put(states_nnet)
    queue2.get()

    start_time = time.time()
    queue1.put(states_nnet)
    print("State nnet send time: %s" % (time.time() - start_time))

    start_time = time.time()
    queue2.get()
    print("States nnet receive time: %.2f" % (time.time() - start_time))

    start_time = time.time()
    proc.join()
    print("Process join time: %.2f" % (time.time() - start_time))


if __name__ == "__main__":
    main()
