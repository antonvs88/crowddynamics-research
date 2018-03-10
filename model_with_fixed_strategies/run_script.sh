#!/bin/bash
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --array=0-99

module load anaconda3
source activate /scratch/work/avonscha/crowd35
python run_script.py 'model_with_fixed_strategies/' $SLURM_ARRAY_TASK_ID
