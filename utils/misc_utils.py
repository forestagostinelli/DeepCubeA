from typing import List, Tuple, Any
import numpy as np


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: List[Any], split_idxs: List[int]) -> List[List[Any]]:
    data_split: List[List[Any]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])

        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split
