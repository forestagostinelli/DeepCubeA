from utils import data_utils, nnet_utils, env_utils
from typing import Dict, List, Tuple, Any

from environments.environment_abstract import Environment
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

    # Gradient Descent
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="Learning rate decay for every iteration. "
                                                                      "Learning rate is decayed according to: "
                                                                      "lr * (lr_d ^ itr)")

    # Training
    parser.add_argument('--max_itrs', type=int, default=1000000, help="Maxmimum number of iterations")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size")
    parser.add_argument('--single_gpu_training', action='store_true',
                        default=False, help="If set, train only on one GPU. Update step will still use "
                                            "all GPUs given by CUDA_VISIBLE_DEVICES")

    # Update
    parser.add_argument('--loss_thresh', type=float, default=0.05, help="When the loss falls below this value, "
                                                                        "the target network is updated.")
    parser.add_argument('--states_per_update', type=int, default=1000, help="How many states to train on before "
                                                                            "checking for update")
    parser.add_argument('--epochs_per_update', type=int, default=1, help="How many epochs to train for. "
                                                                         "Making this greater than 1 could increase "
                                                                         "risk of overfitting, however, one can train "
                                                                         "for more iterations without having to "
                                                                         "generate more data.")
    parser.add_argument('--num_update_procs', type=int, default=1, help="Number of parallel workers used to "
                                                                        "compute updated cost-to-go function")
    parser.add_argument('--update_nnet_batch_size', type=int, default=10000, help="Batch size of each nnet used for "
                                                                                  "each process update. "
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
              num_states: int, heur_fn_i_q, heur_fn_o_qs) -> Tuple[List[np.ndarray], np.ndarray]:
    gbfs_steps: int = min(update_num + 1, num_gbfs_steps)

    # Do updates
    output_time_start = time.time()

    print("Updating cost-to-go with GBFS, with %i step(s)" % gbfs_steps)
    gbfs_updater: GBFSUpdater = GBFSUpdater(env, num_states, data_files, heur_fn_i_q, heur_fn_o_qs,
                                            gbfs_steps, update_batch_size=10000, eps_max=0.0)

    states_update_nnet: List[np.ndarray]
    output_update: np.ndarray
    states_update_nnet, output_update, is_solved = gbfs_updater.update()

    # Print stats
    print("GBFS produced %s states, %.2f%% solved (%.2f seconds)" % (format(output_update.shape[0], ","),
                                                                     100.0 * np.mean(is_solved),
                                                                     time.time() - output_time_start))

    mean_ctg = output_update[:, 0].mean()
    min_ctg = output_update[:, 0].min()
    max_ctg = output_update[:, 0].max()
    print("Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))

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

    # load nnet
    model_save_loc: str = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'])
    model_file: str = "%s/model_state_dict.pt" % model_save_loc

    model_save_loc_prev: str = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'] - 1)
    model_file_prev: str = "%s/model_state_dict.pt" % model_save_loc_prev

    itr: int
    if os.path.isfile(model_file):
        nnet = nnet_utils.load_nnet(model_file, env.get_nnet_model())
        itr = pickle.load(open("%s/train_itr.pkl" % model_save_loc, "rb"))
    elif os.path.isfile(model_file_prev):
        nnet = nnet_utils.load_nnet(model_file_prev, env.get_nnet_model())
        itr = pickle.load(open("%s/train_itr.pkl" % model_save_loc_prev, "rb"))
    else:
        nnet: nn.Module = env.get_nnet_model()
        itr = 0

    nnet.to(device)
    if on_gpu and not args_dict['single_gpu_training']:
        nnet = nn.DataParallel(nnet)

    # training
    num_updates: int = 0

    train_files: List[str] = glob.glob("%s/*.pkl" % args_dict["train_dir"])
    while itr < args_dict['max_itrs']:
        # update
        update_model_dir = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'] - 1)
        all_zeros: bool = args_dict['update_num'] == 0
        heur_fn_i_q, heur_fn_o_qs, heur_procs = nnet_utils.start_heur_fn_runners(args_dict['num_update_procs'],
                                                                                 update_model_dir,
                                                                                 device, on_gpu, env,
                                                                                 all_zeros=all_zeros,
                                                                                 clip_zero=True,
                                                                                 batch_size=args_dict[
                                                                                     "update_nnet_batch_size"])

        states_nnet: List[np.ndarray]
        outputs: np.ndarray
        states_nnet, outputs = do_update(train_files, args_dict['update_num'], env,
                                         args_dict['max_update_gbfs_steps'], args_dict['states_per_update'],
                                         heur_fn_i_q, heur_fn_o_qs)

        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_i_q)

        num_updates += 1

        # train nnet
        num_train_itrs: int = args_dict['epochs_per_update'] * np.ceil(outputs.shape[0] / args_dict['batch_size'])
        print("Training model for update number %i for %i iterations" % (args_dict['update_num'], num_train_itrs))
        last_loss = nnet_utils.train_nnet(nnet, states_nnet, outputs, device, args_dict['batch_size'], num_train_itrs,
                                          itr, args_dict['lr'], args_dict['lr_d'])
        itr += num_train_itrs

        # save nnet
        model_save_loc: str = "%s/%s" % (args_dict['model_dir'], args_dict['update_num'])
        if not os.path.exists(model_save_loc):
            os.makedirs(model_save_loc)

        torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % model_save_loc)
        pickle.dump(itr, open("%s/train_itr.pkl" % model_save_loc, "wb"), protocol=-1)

        # test
        start_time = time.time()
        if num_updates % args_dict['testing_freq'] == 0:
            heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env,
                                                       batch_size=args_dict['update_nnet_batch_size'])
            gbfs_test(args_dict['val_dir'], env, heuristic_fn, max_solve_steps=args_dict['update_num']+1)

        print("Test time: %.2f" % (time.time() - start_time))

        # clear cuda memory
        torch.cuda.empty_cache()

        print("Last loss was %f" % last_loss)
        if last_loss < args_dict['loss_thresh']:
            # Update nnet
            print("Updating target network")
            args_dict['update_num'] = args_dict['update_num'] + 1

    print("Done")


if __name__ == "__main__":
    main()
