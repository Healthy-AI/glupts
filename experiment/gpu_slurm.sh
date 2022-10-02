#!/bin/bash
#SBATCH -J %s
#SBATCH -o %s
#SBATCH -e %s
#SBATCH --cpus-per-task=%d
#SBATCH --gpus-per-node=T4:1
#SBATCH --ntasks=1
#SBATCH --time 0-%d
#SBATCH --time-min 0-%d
cd ~
export GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git
export GIT_PYTHON_REFRESH=quiet
# ACTIVATE CONDA ENV HERE
source somepath/activate MY_ENV
# UPDATE PATHS AS NEEDED
export PATH="somepath/MY_ENV/bin:$PATH"
export PYTHONPATH="$PYTHONPATH:$HOME/Privileged_Times_Series_Learning"

srun python3 Privileged_Times_Series_Learning/experiment/slurm_job.py --config_path "%s" --workers %d --csv_path "%s"
