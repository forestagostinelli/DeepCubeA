# DeepCubeA
This is the code for DeepCubeA. The code is still in development as it is being updated for python3 and PyTorch.

For any issues, please contact Forest Agostinelli (fagostin@uci.edu)

# Setup
For required python packages, please see requirements.txt.
All packages should be able to be installed with pip

Python version used: 3.7.2

Before running anything, please execute: `source setup.sh` in the DeepCubeA directory to add the current directory to your python path.

# Sample script


# Parallelism
Data generation, training, and solving can be easily parallelized across multiple CPUs and GPUs.

When generating data with `scripts/generate_dataset.py`, set the number of CPUs used with `--num_procs`

When training with `ctg_approx/avi.py`, set the number of workers for doing approximate value iteration with `--num_update_procs`

The number of CPUs and GPUs used can be controlled by setting the `CUDA_VISIBLE_DEVICES` environment variable
