#!/bin/bash

#SBATCH --job-name=vedai
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --nodelist=nova[32]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=main
#SBATCH --gres=gpu:1
##SBATCH --mem=20G
#SBATCH --time=30-0
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
module purge
module load cuda/11.7

## conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"

conda activate VisIrNet

##srun nvidia-smi

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
##>>>>>>>>>>>>>>>>>>>>>>>> vedai backbone <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
echo "--**Training Vedai with ssim_pixel **--"
srun nvidia-smi && python Train.py --config-file vedai_default_config.json --b_loss_function ssim_pixel --train_first_stage True 
echo "*************************************************"


echo "--**Training Vedai with mse_pixel **--"
srun nvidia-smi && python Train.py --config-file vedai_default_config.json --b_loss_function mse_pixel  --train_first_stage True
echo "*************************************************"


echo "--**Training Vedai with mae_pixel **--"
srun nvidia-smi && python Train.py --config-file vedai_default_config.json --b_loss_function mae_pixel  --train_first_stage True
echo "*************************************************"


echo "--**Training Vedai with sse_pixel **--"
srun nvidia-smi && python Train.py --config-file vedai_default_config.json --b_loss_function sse_pixel  --train_first_stage True
echo "*************************************************"

## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DATASETS iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## srun python scripts/check_dataset_integrity.py 