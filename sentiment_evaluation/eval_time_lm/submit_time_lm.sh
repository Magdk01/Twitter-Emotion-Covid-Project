#!/bin/sh
#BSUB -J pilot_subset_1
#BSUB -o pilot_subset_1_%J.out
#BSUB -e pilot_subset_1_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
#BSUB -N
#BSUB -B
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment


# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source test-env/bin/activate

python time_lm_masterscript.py