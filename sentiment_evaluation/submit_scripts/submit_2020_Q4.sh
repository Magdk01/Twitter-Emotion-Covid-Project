#!/bin/sh
#BSUB -J pilot_time_lm
#BSUB -o pilot_time_lm_o_%J.out
#BSUB -e pilot_time_lm_o_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
#BSUB -N
#BSUB -B
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment


# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source twitter_venv/bin/activate

python3 "sentiment_eval_time_lm.py" "data/quarterly_data_2020_Q4" "cardiffnlp/twitter-roberta-base-dec2020"
