#!/bin/bash

#SBATCH --job-name=sky__mae_sse__ssim_mse
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --nodelist=nova[83]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=main
#SBATCH --gres=gpu:1
##SBATCH --mem=20G
#SBATCH --time=15-0
#SBATCH --output=slurm_logs/%x-%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alain.ndigande@ozu.edu.tr


echo "---- env ! ----"

## ulimit -s unlimited
## ulimit -l unlimited
## ulimit -a

echo "------- setup done ! -----"
## Load the python interpreter
##clear the module
# module purge
# module load cuda/11.7

## conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"

conda activate VisIrNet

##srun nvidia-smi

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRAINING iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

echo "--**Training**--"

##>>>>>>>>>>>>>>>>>>>>>>>> skydata backbone <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --b_loss_function ssim_pixel --train_first_stage True 
# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --b_loss_function mse_pixel  --train_first_stage True
# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --b_loss_function mae_pixel  --train_first_stage True
# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --b_loss_function sse_pixel  --train_first_stage True

##### second stage
## "l1_homography_loss" ,"l2_homography_loss" , "l1_corners_loss" , "l2_corners_loss" 

# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --r_loss_function l2_corners_loss --b_loss_function ssim_pixel --train_second_stage True 
# srun nvidia-smi && python Train.py --config-file skydata_default_config.json --r_loss_function l2_corners_loss --b_loss_function mse_pixel  --train_second_stage True
srun nvidia-smi && python Train.py --config-file skydata_default_config.json --r_loss_function l2_corners_loss --b_loss_function mae_pixel  --train_second_stage True
srun nvidia-smi && python Train.py --config-file skydata_default_config.json --r_loss_function l2_corners_loss --b_loss_function sse_pixel  --train_second_stage True

## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DATASETS iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## srun python scripts/check_dataset_integrity.py 