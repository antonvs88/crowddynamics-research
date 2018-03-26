#! /bin/bash
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --mem-per-cpu=2500
#SBATCH --array=0-99

module load anaconda3
source activate /scratch/work/avonscha/crowd35
python calculate_field_data.py $SLURM_ARRAY_TASK_ID
