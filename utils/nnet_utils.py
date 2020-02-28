from typing import List, Tuple
import numpy as np
import os
import torch
from torch import nn
from environments.environment_abstract import Environment
from collections import OrderedDict
import re
from random import choice
from torch import Tensor

import torch.optim as optim
from torch.optim.optimizer import Optimizer

import time


# training
def states_nnet_to_pytorch_input(states_nnet: List[List], device) -> List[Tensor]:
    states_nnet_tensors = []
    for tensor_idx in range(len(states_nnet[0])):
        tensor_np = np.stack([x[tensor_idx] for x in states_nnet])
        tensor = torch.tensor(tensor_np, device=device)

        states_nnet_tensors.append(tensor)

    return states_nnet_tensors


def make_batches(data: Tuple[List, np.ndarray], batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rand_idxs = np.random.choice(len(data[0]), len(data[0]), replace=False)
    inputs_all: List = [data[0][i] for i in rand_idxs]
    outputs_all = data[1][rand_idxs].astype(np.float32)
    start_idx = 0
    batches = []
    while (start_idx + batch_size) <= len(inputs_all):
        end_idx = start_idx + batch_size

        inputs_batch = inputs_all[start_idx:end_idx]
        outputs_batch = outputs_all[start_idx:end_idx]

        batches.append((inputs_batch, outputs_batch))

        start_idx = end_idx

    return batches


def train_nnet(nnet: nn.Module, data: Tuple[List, np.ndarray], device: torch.device, on_gpu: bool,
               batch_size: int, num_itrs: int, train_itr: int = 0, display: bool = True):
    # optimization
    display_itrs = 100
    criterion = nn.MSELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=0.001)
    # optimizer: Optimizer = env.get_optimizer()

    # initialize status tracking
    start_time = time.time()

    # train network
    batches: List[Tuple[np.ndarray, np.ndarray]] = make_batches(data, batch_size)

    nnet.train()
    max_itrs: int = train_itr + num_itrs
    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()

        # get data
        inputs_batch_np, targets_batch_np = choice(batches)
        targets_batch_np = targets_batch_np.astype(np.float32)

        # send data to device
        states_batch: List[Tensor] = states_nnet_to_pytorch_input(inputs_batch_np, device)
        targets_batch: Tensor = torch.tensor(targets_batch_np, device=device)

        # forward
        nnet_outputs_batch: Tensor = nnet(*states_batch)

        # cost
        nnet_cost_to_go = nnet_outputs_batch[:, 0]
        target_cost_to_go = targets_batch[:, 0]

        loss = criterion(nnet_cost_to_go, target_cost_to_go)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if (train_itr % display_itrs == 0) and display:
            print("Itr: %i, loss: %.2f, targ_ctg: %.2f, nnet_ctg: %.2f, "
                  "Time: %.2f" % (
                      train_itr, loss.item(), target_cost_to_go.mean().item(), nnet_cost_to_go.mean().item(),
                      time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1


# pytorch device
def get_device() -> Tuple[torch.device, List[int], bool]:
    device: torch.device = torch.device("cpu")
    devices: List[int] = []
    on_gpu: bool = False
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and torch.cuda.is_available():
        device = torch.device("cuda:%i" % 0)
        devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
        on_gpu = True

    return device, devices, on_gpu


# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


# heuristic
def get_heuristic_fn(nnet: nn.Module, device: torch.device, env: Environment, clip_zero: bool = False,
                     batch_size: int = 5000):
    nnet.eval()

    def heuristic_fn(states: List) -> np.ndarray:
        cost_to_go: np.ndarray = np.zeros(0)

        num_states: int = len(states)
        start_idx: int = 0
        while start_idx < num_states:
            # get batch
            end_idx: int = min(start_idx + batch_size, num_states)
            states_batch: List = states[start_idx:end_idx]

            # convert to nnet input
            states_nnet_batch: List[List] = env.state_to_nnet_input(states_batch)

            # get nnet output
            states_nnet_batch_tensors = states_nnet_to_pytorch_input(states_nnet_batch, device)
            cost_to_go_batch: np.ndarray = nnet(*states_nnet_batch_tensors).cpu().data.numpy()

            cost_to_go: np.ndarray = np.concatenate((cost_to_go, cost_to_go_batch[:, 0]), axis=0)

            start_idx: int = end_idx

        assert (cost_to_go.shape[0] == num_states)

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def get_available_gpu_nums() -> List[int]:
    gpu_nums: List[int] = []
    if ('CUDA_VISIBLE_DEVICES' in os.environ) and (len(os.environ['CUDA_VISIBLE_DEVICES']) > 0):
        gpu_nums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]

    return gpu_nums


def load_heuristic_fn(nnet_dir: str, device: torch.device, on_gpu: bool, nnet: nn.Module, env: Environment,
                      clip_zero: bool = False, gpu_num: int = -1, batch_size: int = 1000):
    if (gpu_num >= 0) and on_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    model_file = "%s/model_state_dict.pt" % nnet_dir

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn = get_heuristic_fn(nnet, device, env, clip_zero=clip_zero, batch_size=batch_size)

    return heuristic_fn
