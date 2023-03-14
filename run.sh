#!/bin/bash
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --partition=compsci-gpu

rm -rf ~/.cache/torch_extensions

start=`date +%s`

set -ex

python3 ctg_approx/avi.py --env cube3 --states_per_update 50000000 --batch_size 10000 --nnet_name cube3 --max_itrs 2000000 --loss_thresh 0.06 --back_max 30 --num_update_procs 39

end=`date +%s`
runtime=$((end-start))
