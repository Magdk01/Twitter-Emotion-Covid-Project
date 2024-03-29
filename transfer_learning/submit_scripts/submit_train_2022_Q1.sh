#!/bin/sh
#BSUB -J train_2022_Q1
#BSUB -o train_2022_Q1_%J.out
#BSUB -e train_2022_Q1_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -N
#BSUB -B
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment


# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source  twitter_venv/bin/activate

python Quarterly_trainer.py "twitter-roberta-base-mar2022" "2022_Q1"  
