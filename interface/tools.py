import sys
import argparse
sys.path.append('../')
from search_methods import astar
from environments.cube3 import Cube3State
from utils import env_utils
import numpy as np

def prepare_args():
    # parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--states', type=str, default='', help="File containing states to solve")
    parser.add_argument('--model_dir', type=str, default='../saved_models/cube3/current/', help="Directory of nnet model")
    parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--batch_size', type=int, default=3000, help="Batch size for BWAS")
    parser.add_argument('--weight', type=float, default=0.3, help="Weight of path cost")
    parser.add_argument('--language', type=str, default="python", help="python or cpp")

    parser.add_argument('--results_dir', type=str, default='', help="Directory to save results")
    parser.add_argument('--start_idx', type=int, default=0, help="")
    parser.add_argument('--nnet_batch_size', type=int, default=36000, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect final results, "
                                                                          "but will help if nnet is running out of "
                                                                          "memory.")

    parser.add_argument('--verbose', action='store_true', default=True, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging")

    args = parser.parse_args([])
    return args

def getResults(state):
    args = prepare_args()
    env = env_utils.get_environment(args.env)

    # convert
    FEToState = [6, 3, 0, 7, 4, 1, 8, 5, 2, 15, 12, \
                       9, 16, 13, 10, 17, 14, 11, 24, 21, 18, \
                       25, 22, 19, 26, 23, 20, 33, 30, 27, 34, \
                       31, 28, 35, 32, 29, 38, 41, 44, 37, 40, \
                       43, 36, 39, 42, 51, 48, 45, 52, 49, 46, \
                       53, 50, 47]
    converted_state = []
    for i in range(len(FEToState)):
        converted_state.append(state[FEToState[i]])

    ### Load starting states
    state = np.array(converted_state, np.int64)

    states = []
    states.append(Cube3State(state))
    # call bwas_python
    solns, paths, times, num_nodes_gen = astar.bwas_python(args, env, states)

    # moves string
    action_map = [(f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

    moves = []
    moves_rev = []
    solve_text = []

    for action in solns[0]:
        i = action_map[action]
        moves.append(str(i[0]) + '_' + str(i[1]))
        moves_rev.append(str(i[0]) + '_' + str(-i[1]))
        if i[1] == -1:
            solve_text.append(str(i[0]) + "'")
        else:
            solve_text.append(str(i[0]))

    results = {"moves": moves, "moves_rev": moves_rev, "solve_text": solve_text}
    return results



if __name__ == "__main__":
    # arr = [ 8, 10, 36,  3,  4, 12, 27, 34, 53,  9,  5,  0, 46, 13, 19, 11, 30,
    #         2, 18, 52, 15, 41, 22, 21, 47,  1, 35, 44,  7, 17, 16, 31, 37, 24,
    #        48, 29, 45, 14,  6, 25, 40, 39, 42, 28, 33, 26, 23, 38, 43, 49, 50,
    #        20, 32, 51]

    arr = [ 51, 32, 26, 30, 4, 3, 2, 19, 36, 9, 39, 29, 28, 13, 14, 38, 5, 45, 
        27, 16, 44, 21, 22, 46, 8, 52, 42, 15, 50, 47, 23, 31, 34, 6, 48, 11, 35, 
        41, 24, 10, 40, 37, 17, 7, 0, 20, 43, 33, 25, 49, 1, 18, 12, 53]

    print(arr)
    print(getResults(arr))
