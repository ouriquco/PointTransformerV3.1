#!/bin/bash
#SBATCH --job-name=depth_testing
#SBATCH --output=./slurm_exp_logs/output/depth-exp_testing.out
#SBATCH --error=./slurm_exp_logs/errors/depth_testing.err
#SBATCH --partition=condo
#SBATCH --nodelist=condo7

# Choose correct MIG partition
# export CUDA_VISIBLE_DEVICES=MIG-f6cb9f47-ec7c-558c-b9fc-c57c52ef1ad8

#Initialize Conda
source /home/010892622/miniconda3/etc/profile.d/conda.sh

# set up the environment
conda activate pointcept_V3

# training script
cd /data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1
python inference.py
# sh scripts/train.sh -p python -g 1 -d waymo -c semseg-pt-v3m1-0-base_depth_exp -n semsegv3-waymo-depth-exp2
