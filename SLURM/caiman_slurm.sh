#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --no-requeue                 # Do not re-run job  if preempted
#SBATCH --job-name=caiman_elena      # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=50           # Number of requested cores
#SBATCH --mem=50000                  # Real memory (RAM) required (MB)
#SBATCH --time=20:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export your current env to the job env

WORKING_DIRECTORY='/home/tjv55/elena_tbi' # !!! You need to change this to the parent directory
cd $WORKING_DIRECTORY  # go to the parent directory
source activate caiman #Activate your anaconda environment you have installed

#THE MAGIC HAPPENS HERE
#srun is for slurm, then we use "find" to search current and subdirectories for tif files
#then we "execdir" within "find" so that on each file it finds it runs the caiman pipeline and saves
#the data back in its respective directory, "{}" represents the filename to be loaded.
srun find . -type f -name *.tif -execdir python /home/tomi/CaImAn/SLURM/slurm_pipeline.py {} \;
