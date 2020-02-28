# Set GPU
from typing import List, Tuple

import numpy as np
import pickle

import sys

from random import choice


class Logger(object):
    def __init__(self, filename: str, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_states_from_files(num_states: int, data_files: List[str],
                           load_outputs: bool = False) -> Tuple[List, np.ndarray]:
    states = []
    outputs_l = []
    while len(states) < num_states:
        data_file = choice(data_files)
        data = pickle.load(open(data_file, "rb"))

        rand_idxs = np.random.choice(len(data['states']), len(data['states']), replace=False)
        num_samps: int = min(num_states - len(states), len(data['states']))

        for idx in range(num_samps):
            rand_idx = rand_idxs[idx]
            states.append(data['states'][rand_idx])

        if load_outputs:
            for idx in range(num_samps):
                rand_idx = rand_idxs[idx]
                outputs_l.append(data['outputs'][rand_idx])

    outputs = np.array(outputs_l)
    outputs = np.expand_dims(outputs, 1)

    return states, outputs
