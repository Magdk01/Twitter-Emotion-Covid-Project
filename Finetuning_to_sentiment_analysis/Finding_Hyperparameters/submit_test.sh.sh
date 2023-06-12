#!/bin/sh
#BSUB -J test
#BSUB -o test%J.out
#BSUB -e test%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 1
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 18:00
#BSUB -N
#BSUB -B
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.6.3-python-3.9.6
module load cuda/11.7

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source FRANK/bin/activate

python test.py
