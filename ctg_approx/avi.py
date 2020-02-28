from utils import data_utils, nnet_utils, env_utils
from typing import Dict, List, Tuple, Any

from environments.environment_abstract import Environment, State
from updaters.gbfs_updater import GBFSUpdater
from search_methods.gbfs import gbfs_test
import torch

import torch.nn as nn
import os
import pickle

from argparse import ArgumentParser
import numpy as np
import time

import sys
import glob


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment")

    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help="")

    # Training
    parser.add_argument('--max_updates', type=int, default=1000, help="Maxmimum number of updates")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size")
    parser.add_argument('--single_gpu_training', action='store_true',
                        default=False, help="If set, train only on one GPU. Update step will still use "
                                            "all GPUs given by CUDA_VISIBLE_DEVICES")

    # Update
    parser.add_argument('--states_per_update', type=int, default=1000, help="How many states to load per update")
    parser.add_argument('--epochs_per_update', type=int, default=1, help="How many epochs to train for each update")
    parser.add_argument('--num_update_procs', type=int, default=1, help="Number of parallel workers used to "
                                                                        "compute updated cost-to-go function")
    parser.add_argument('--update_nnet_batch_size', type=int, default=1000, help="Batch size of each nnet used for each"
                                                                                 " process update. "
                                                                                 "Make smaller if running out of "
                                                                                 "memory.")
    parser.add_argument('--max_update_gbfs_steps', type=int, default=1, help="Number of steps to take when trying to "
                                                                             "solve training states with "
                                                                             "greedy best-first search (GBFS). "
                                                                             "Each state "
                                                                             "encountered when solving is added to the "
                                                                             "training set. Number of steps starts at "
                                                                             "1 and is increased every update until "
                                                                             "the maximum number is reached. "
                                                                             "Value of 1 is the same as doing "
                                                                             "value iteration on only given training "
                                                                             "states. Increasing this number "
                                                                             "can make the cost-to-go function more "
                                                                             "robust by exploring more of the "
                                                                             "state space.")
    # Testing
    parser.add_argument('--testing_freq', type=int, default=1, help="How frequently (i.e. after how many updates)"
                                                                    " to test with GBFS.")
    parser.add_argument('--testing_solve_steps', type=int, default=None, help="Maximum number of steps to take when "
                                                                              "solving with GBFS. If not given, "
                                                                              "then this is set to the maximum number "
                                                                              "of backwards steps taken to create the "
                                                                              "validation data")

    # data
    parser.add_argument('--train_dir', type=str, required=True, help="Directory of training data")
    parser.add_argument('--val_dir', type=str, required=True, help="Directory of validation data")

    # model
    parser.add_argument('--nnet_name', type=str, required=True, help="Name of neural network")
    parser.add_argument('--update_num', type=int, default=0, help="Update number")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Director to which to save model")

    # parse arguments
    args = parser.parse_args()

    args_dict: Dict[str, Any] = vars(args)

    # make save directory
    args_dict['model_dir'] = "%s/%s/" % (args_dict['save_dir'], args_dict['nnet_name'])
    model_save_loc = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'])
    if not os.path.exists(model_save_loc):
        os.makedirs(model_save_loc)

    args_dict["output_save_loc"] = "%s/%s/output.txt" % (args_dict['save_dir'], args_dict['nnet_name'])

    # save args
    args_save_loc = "%s/args.pkl" % model_save_loc
    print("Saving arguments to %s" % model_save_loc)
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=-1)

    print("Batch size: %i" % args_dict['batch_size'])

    return args_dict


def do_update(data_files: List[str], update_num: int, env: Environment, num_gbfs_steps: int,
              num_states: int, num_procs: int, nnet_dir: str, device, on_gpu: bool,
              nnet_batch_size: int) -> Tuple[List, np.ndarray]:
    all_zeros: bool = update_num == 0

    gbfs_steps: int = min(update_num + 1, num_gbfs_steps)

    # Get data
    data_time_start = time.time()
    states: List[State]
    states, _ = data_utils.load_states_from_files(num_states, data_files)

    print("Loaded %s datapoints (%.2f seconds)" % (format(len(states), ","), time.time() - data_time_start))

    # Do updates
    output_time_start = time.time()

    print("Updating cost-to-go with GBFS, with %i step(s)" % gbfs_steps)
    gbfs_updater: GBFSUpdater = GBFSUpdater(env, all_zeros, num_procs, nnet_dir, device, on_gpu, nnet_batch_size,
                                            gbfs_steps, search_batch_size_max=1000, eps_max=0.1)

    states_update: List[State]
    output_update: np.ndarray
    states_update, output_update, is_solved = gbfs_updater.update(states, verbose=True)

    gbfs_updater.cleanup()

    # Print stats
    print("GBFS produced %s states, %.2f%% solved (%.2f seconds)" % (format(output_update.shape[0], ","),
                                                                     100.0 * np.mean(is_solved),
                                                                     time.time() - output_time_start))

    mean_ctg = output_update[:, 0].mean()
    min_ctg = output_update[:, 0].min()
    max_ctg = output_update[:, 0].max()
    print("Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))

    states_update_nnet = env.state_to_nnet_input(states_update)

    return states_update_nnet, output_update


def main():
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    if not args_dict["debug"]:
        sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")

    # environment
    env: Environment = env_utils.get_environment(args_dict['env'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # training
    itr = 0
    num_updates: int = 0

    train_files: List[str] = glob.glob("%s/*.pkl" % args_dict["train_dir"])
    while args_dict['update_num'] < args_dict['max_updates']:
        # update
        update_model_dir = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'] - 1)
        states_outputs: Tuple[List, np.ndarray]
        states_outputs = do_update(train_files, args_dict['update_num'], env,
                                   args_dict['max_update_gbfs_steps'], args_dict['states_per_update'],
                                   args_dict['num_update_procs'], update_model_dir, device, on_gpu,
                                   args_dict['update_nnet_batch_size'])

        num_updates += 1

        # load nnet
        start_time = time.time()
        model_save_loc = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'])
        model_file = "%s/model_state_dict.pt" % model_save_loc
        if os.path.isfile(model_file):
            nnet = nnet_utils.load_nnet(model_file, env.get_nnet_model())
        else:
            nnet: nn.Module = env.get_nnet_model()

        nnet.to(device)
        if on_gpu and not args_dict['single_gpu_training']:
            nnet = nn.DataParallel(nnet)
        print("Load nnet time: %s" % (time.time() - start_time))

        # train nnet
        num_train_itrs: int = args_dict['epochs_per_update']*np.ceil(len(states_outputs[0]) / args_dict['batch_size'])
        print("Training model for update number %i for %i iterations" % (args_dict['update_num'], num_train_itrs))
        nnet_utils.train_nnet(nnet, states_outputs, device, on_gpu, args_dict['batch_size'], num_train_itrs,
                              train_itr=itr)
        itr += num_train_itrs

        if not os.path.exists(model_save_loc):
            os.makedirs(model_save_loc)

        # save nnet
        torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)

        # test
        if num_updates % args_dict['testing_freq'] == 0:
            heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env)
            gbfs_test(args_dict['val_dir'], env, heuristic_fn, max_solve_steps=args_dict['testing_solve_steps'])

        # clear cuda memory
        del nnet
        torch.cuda.empty_cache()

        args_dict['update_num'] = args_dict['update_num'] + 1

    print("Done")


if __name__ == "__main__":
    main()
