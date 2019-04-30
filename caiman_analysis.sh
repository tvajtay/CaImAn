#!/bin/bash

conda activate caiman #Activate your anaconda environment you have installed

#Environment variables to make sure CaImAn runs efficiently
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#THE MAGIC HAPPENS HERE
#srun is for slurm, then we use "find" to search current and subdirectories for tif files
#then we "execdir" within "find" so that on each file it finds it runs the caiman pipeline and saves
#the data back in its respective directory, "{}" represents the filename to be loaded.
find . -type f -name *.hdf5 -execdir python /home/tomi/CaImAn/H5_pipeline.py {} \;

#find . -type f -name *.tif -execdir python /home/tomi/CaImAn/SLURM/slurm_pipeline.py {} \;
