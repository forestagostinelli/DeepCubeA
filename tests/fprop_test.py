from typing import List
from argparse import ArgumentParser
import torch
import torch.nn as nn

from environments.environment_abstract import Environment, State
from utils import nnet_utils
from utils import env_utils

import time


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--num_states', type=int, default=100, help="")
    parser.add_argument('--back_max', type=int, default=30, help="")

    args = parser.parse_args()

    # get environment
    env: Environment = env_utils.get_environment(args.env)

    # get heuristic fn
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    nnet: nn.Module = env.get_nnet_model()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env)

    # load file
    start_time = time.time()
    states: List[State] = env.generate_states(args.num_states, (0, args.back_max))

    state_gen_time = time.time() - start_time
    states_gen_per_sec = len(states)/state_gen_time
    print("Generated %i states in %s seconds (%.2f/second)" % (len(states), state_gen_time, states_gen_per_sec))

    # initialize
    heuristic_fn(states)

    # compute
    start_time = time.time()
    heuristic_fn(states)

    nnet_time = time.time() - start_time
    states_per_sec = len(states)/nnet_time
    print("Computed heuristic for %i states in %s seconds (%.2f/second)" % (len(states), nnet_time, states_per_sec))


if __name__ == "__main__":
    main()
