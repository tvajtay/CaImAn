#!/bin/bash

#Shell script used to submit a SLURM job with SBATCH
#You would call this shell script with SBATCH
#"sbatch /path/to/script/caiman_slurm.sh"


#SBATCH --partition=main             # Partition (job queue)
#SBATCH --no-requeue                 # Do not re-run job  if preempted
#SBATCH --job-name=caiman_naomi      # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=15           # Number of requested CPU's (Max @ Rutgers 30/node)
#SBATCH --mem=60000                  # Real memory (RAM) required (MB) (Max @ Rutgers is 120000/node)
#SBATCH --time=30:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export your current env to the job env, not neccessary but good to know

WORKING_DIRECTORY='/scratch/tjv55/data' # !!! You need to change this to the parent directory of files to be analyzed
cd $WORKING_DIRECTORY  # go to the parent directory
source activate caiman #Activate your anaconda environment you have installed previously on the HPC

#Declaring environment variables to make sure CaImAn runs efficiently
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#THE MAGIC HAPPENS HERE
#First we use the internal BASH command "find" to search current and subdirectories for tif files
#then we "execdir" within "find" so that on each file it finds it runs the caiman pipeline and saves
#the data back in its respective directory, "{}" represents the filename to be loaded.
#'slurm_pipeline.py' takes a single CLI argument which is the file to be analyzed.
find -type f -name *.tif -execdir python /home/tjv55/slurm_pipeline.py {} \;

#The following would be the variation of the previous command if you didn't want to include
#the parent directory itself and search just the sub-directories.

#find . -type f -name *.tif -execdir python /home/tomi/CaImAn/SLURM/slurm_pipeline.py {} \;
