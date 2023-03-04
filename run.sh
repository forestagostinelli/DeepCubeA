#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

rm -rf ~/.cache/torch_extensions

start=`date +%s`

set -ex

python3 latent_inversion.py

end=`date +%s`
runtime=$((end-start))
