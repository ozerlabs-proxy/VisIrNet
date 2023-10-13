#!/bin/bash

#SBATCH --job-name=Vis
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
module purge
module load cuda/11.7

## conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"

conda activate VisIrNet

##srun nvidia-smi

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TRAINING iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

echo "--**Training**--"
# # srun python Train.py --config-file skydata_default_config.json
srun nvidia-smi && python3 Train.py --config-file vedai_default_config.json
# # srun nvidia-smi && python3 Train.py --config-file googlemap_default_config.json 



#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<DATASETS iNTEGRITY>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# srun python scripts/check_dataset_integrity.py 