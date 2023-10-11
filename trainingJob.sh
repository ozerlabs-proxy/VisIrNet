#!/bin/bash

#SBATCH --job-name=VisIrNet-skydata
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --nodelist=nova[83]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --gres=gpu:1
##SBATCH --mem=20G
#SBATCH --time=15-0
#SBATCH --output=slurm_logs/%x-%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alain.ndigande@ozu.edu.tr


echo "---- setting up env ! ----"

ulimit -s unlimited
ulimit -l unlimited
ulimit -a

echo "------- setup done ! -----"
## Load the python interpreter
#clear the module
module purge

# conda environment
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"

conda activate VisIrNet

srun nvidia-smi

echo "------ Training -------"
srun bash training.sh
