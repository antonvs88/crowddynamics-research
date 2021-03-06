#! /bin/bash
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=2500
#SBATCH --array=0-49

module load anaconda3
source activate /scratch/work/avonscha/crowd35
python hdf5_to_npy_gz.py $SLURM_ARRAY_TASK_ID
