#!/bin/bash

#SBATCH --job-name=TrInfJob
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --nodelist=nova[82]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=main
#SBATCH --gres=gpu:1
##SBATCH --mem=80G
#SBATCH --time=15-0
#SBATCH --output=slurm_logs/%x-%j.txt

echo "---- env ! ----"

ulimit -s unlimited
ulimit -l unlimited
ulimit -a

echo "-----modules -----"
## Load the python interpreter
##clear the module
# module purge
# module load cuda/11.7

## conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"

conda activate VisIrNet

echo "--**Training**--"

## BACKBONE
# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --train_first_stage True

## REGRESSION BLOCK
# "l1_homography_loss" ,"l2_homography_loss" , "l1_corners_loss" , "l2_corners_loss" 
srun nvidia-smi && python Train.py --config-file skydata_default_config.json --train_second_stage True
