# DeepCubeA
This is the code for [DeepCubeA](https://www.ics.uci.edu/~fagostin/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf) for python3 and PyTorch.
The original python2, tensorflow code can be found on [CodeOcean](https://codeocean.com/capsule/5723040/tree/v1).

This code is still under development, however, one should be able to use the DeepCubeA algorithm on the Rubik's cube as well as use DeepCubeA on their own environment.

For any issues, please contact Forest Agostinelli (fagostin@uci.edu)

# Setup
For required python packages, please see requirements.txt.
All packages should be able to be installed with pip

Python version used: 3.7.2

IMPORTANT! Before running anything, please execute: `source setup.sh` in the DeepCubeA directory to add the current directory to your python path.

For the first run, make sure you create the following directories:

`mkdir data`

`mkdir saved_models`

`mkdir results`

`mkdir results/cube3/`

# Sample script
Here are the commands to quickly train DeepCubeA to solve the Rubik's cube.
###### Generate training data
`python scripts/generate_dataset.py --env cube3 --back_max 30 --data_dir data/cube3/train/ --num_per_file 1000000 --num_files 100 --num_procs 1`

###### Generate validation data
`python scripts/generate_dataset.py --env cube3 --back_max 30 --data_dir data/cube3/val/ --num_per_file 10000 --num_files 1 --num_procs 1`

###### Train cost-to-go function
`python ctg_approx/avi.py --env cube3 --states_per_update 1000000 --batch_size 1000 --nnet_name cube3 --num_update_procs 1 --train_dir data/cube3/train/ --val_dir data/cube3/val/ --update_num 0 --max_updates 20`

###### Solve with A* search, use --verbose for more information
`python search_methods/astar.py --states data/cube3/test/data_0.pkl --model saved_models/cube3/19/ --env cube3 --weight 0.1 --batch_size 100 --results_file results/cube3/results.pkl`

###### Solve with greedy best-first search (GBFS)
`python search_methods/gbfs.py --model_dir saved_models/cube3/19/ --env cube3 --data_dir data/cube3/val/ --max_steps 30`

###### Improving Results
During approximate value iteration (AVI), one can get better results by increasing the batch size (`--batch_size`) and number of states per update (`--states_per_update`). Increasing the number of updates (`--max_updates`) can also help.

One can also add additional states to training set by doing GBFS during the update stage and adding the states encountered during GBFS to the states used for approximate value iteration (`--max_update_gbfs_steps`). Setting `--max_update_gbfs_steps` to 1 is the same as doing approximate value iteration.

During A* search, increasing the weight on the path cost (`--weight`, range should be [0,1]) and the batch size (`--batch_size`) generally improves results.

# Using DeepCubeA to Solve Your Own Problem
Create your own environment by implementing the abstract methods in `environments/environment_abstract.py`
See `environments/cube3.py` for an example.

After implementing your method, edit `utils/env_utils.py` to return your environment object given your choosen keyword.

Use `tests/fprop_test.py` to make sure basic aspects of your implementation are working correctly.

# Parallelism
Data generation, training, and solving can be easily parallelized across multiple CPUs and GPUs.

When generating data with `scripts/generate_dataset.py`, set the number of CPUs with `--num_procs`

When training with `ctg_approx/avi.py`, set the number of workers for doing approximate value iteration with `--num_update_procs`
This will spawn a separate DNN for the update step for each worker, so marke sure you have enough memory.
If multiple GPUs are used, then this will spread the workers across the GPUs.

The number of GPUs used can be controlled by setting the `CUDA_VISIBLE_DEVICES` environment variable.

# Compiling c++
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
