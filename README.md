# DeepCubeA
This is the code for [DeepCubeA](https://www.ics.uci.edu/~fagostin/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf) for python3 and PyTorch.
The original python2, tensorflow code can be found on [CodeOcean](https://codeocean.com/capsule/5723040/tree/v1).

This currently contains the code for using DeepCubeA to solve the Rubik's cube, 15-puzzle, 24-puzzle, 35-puzzle, 48-puzzle, and Lights Out.
You can also adapt this code to use DeepCubeA to solve new problems that you might be working on.

For any issues, please contact Forest Agostinelli (foresta@cse.sc.edu)

# Setup
For required python packages, please see requirements.txt.
You should be able to install these packages with pip or conda

Python version used: 3.7.2

IMPORTANT! Before running anything, please execute: `source setup.sh` in the DeepCubeA directory to add the current 
directory to your python path.

# Training and A* Search
`train.sh` contains the commands to trian the cost-to-go function as well as using it with A* search.
Note that some of the hyperparameters may be slightly different than those in the paper as they were later 
found to give slightly better results.

There are pre-trained models in the `saved_models/` directory as well as `output.txt` files to let you know what output to expect.

These models were trained with 1-4 GPUs and 20-30 CPUs. This varies throughout training as the training is often stopped and started again to make room for other processes.

There are pre-computed results of A* search in the `results/` directory. 

### Commands to train DeepCubeA to solve the 15-puzzle.
###### Train cost-to-go function
`python ctg_approx/avi.py --env puzzle15 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle15 --max_itrs 1000000 --loss_thresh 0.1 --back_max 500 --num_update_procs 30`

###### Solve with A* search, use --verbose for more information
`python search_methods/astar.py --states data/puzzle15/test/data_0.pkl --model saved_models/puzzle15/current/ --env puzzle15 --weight 0.8 --batch_size 20000 --results_dir results/puzzle15/ --language cpp --nnet_batch_size 10000`

###### Compare to shortest path
`python scripts/compare_solutions.py --soln1 data/puzzle15/test/data_0.pkl --soln2 results/puzzle15/results.pkl`

### Improving Results
During approximate value iteration (AVI), one can get better results by increasing the batch size (`--batch_size`) and number of states per update (`--states_per_update`).
Decreasing the threshold before the target network is updated (`--loss_thresh`) can also help.

One can also add additional states to training set by doing greedy best-first search (GBFS) during the update stage and adding the states encountered during GBFS to the states used for approximate value iteration (`--max_update_steps`). Setting `--max_update_steps` to 1 is the same as doing approximate value iteration.

During A* search, increasing the weight on the path cost (`--weight`, range should be [0,1]) and the batch size (`--batch_size`) generally improves results.

These improvements often come at the expense of time.

# Using DeepCubeA to Solve New Problems
Create your own environment by implementing the abstract methods in `environments/environment_abstract.py`
See the implementations in `environments/` for examples.

After implementing your method, edit `utils/env_utils.py` to return your environment object given your chosen keyword.

Use `tests/timing_test.py` to make sure basic aspects of your implementation are working correctly.

# Parallelism
Training and solving can be easily parallelized across multiple CPUs and GPUs.

When training with `ctg_approx/avi.py`, set the number of workers for doing approximate value iteration with `--num_update_procs`
During the update process, the target DNN is spawned on each available GPU and they work in parallel during the udpate step.

The number of GPUs used can be controlled by setting the `CUDA_VISIBLE_DEVICES` environment variable.

i.e. `export CUDA_VISIBLE_DEVICES="0,1,2,3"`

# Memory
When obtaining training data with approximate value iteration and solving using A* search, the batch size of the data 
given to the DNN can be controlled with `--update_nnet_batch_size` for the `avi.py` file and `--nnet_batch_size` for
the `astar.py` file. Reduce this value if your GPUs are running out of memory during approximate value iteration or 
during A* search.

# Compiling C++ for A* Search
`cd cpp/`

`make`

If you are not able to get the C++ version working on your computer, you can change the `--language` switch for
`search_methods/astar.py` from `--language cpp` to `--language python`.
Note that the C++ version is generally faster.
