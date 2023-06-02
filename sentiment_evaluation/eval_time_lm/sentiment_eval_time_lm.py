from transformers import pipeline
from transformers import pipeline
import pandas as pd
import tqdm
import os
import torch
import sys

def sentiment_for_df(text):
    classification = sentiment_task(text)[0]['label']

    if classification =='negative':
        return -1
    elif classification =='neutral':
        return 0
    elif classification =='positive':
        return 1

"""
For the "MODEL" parameter use one of:
    cardiffnlp/twitter-roberta-base-sentiment-latest
    cardiffnlp/twitter-roberta-base-sentiment
    
    cardiffnlp/twitter-roberta-base-2019-90m
    cardiffnlp/twitter-roberta-base-mar2020
    cardiffnlp/twitter-roberta-base-jun2020
    cardiffnlp/twitter-roberta-base-sep2020	
    cardiffnlp/twitter-roberta-base-dec2020	
    cardiffnlp/twitter-roberta-base-mar2021	
    cardiffnlp/twitter-roberta-base-jun2021
    cardiffnlp/twitter-roberta-base-sep2021
    cardiffnlp/twitter-roberta-base-dec2021
    cardiffnlp/twitter-roberta-base-2021-124m
    cardiffnlp/twitter-roberta-base-mar2022
    cardiffnlp/twitter-roberta-base-jun2022
    cardiffnlp/twitter-roberta-base-mar2022-15M-incr
    cardiffnlp/twitter-roberta-base-jun2022-15M-incr
    cardiffnlp/twitter-roberta-base-sep2022
    cardiffnlp/twitter-roberta-base-sep2022
    cardiffnlp/twitter-roberta-base-2022-154m	
    cardiffnlp/twitter-roberta-large-2022-154m	
    
"""

#Sets the cache folder for the transformer models on the HPC's HOME drive
os.environ['TRANSFORMERS_CACHE'] = './cache/'

DATASET = sys.argv[1]
MODEL = sys.argv[2] # See above for example

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'Using Cuda device: {device}\n')
else:    
    raise RuntimeError('No CUDA available')

assert len(sys.argv) == 3, 'Script needs a upper and lower limit'

lower_limit = int(sys.argv[1])
upper_limit = int(sys.argv[2])

assert lower_limit < upper_limit, 'First limit must be the lowest'

print(f'Will look in range:\n{lower_limit}:{upper_limit}')

subset_range = (lower_limit,upper_limit)
tqdm.tqdm.pandas()

sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL,device=0)
print('pipeline loaded')

print('reading dataset...')
df = pd.read_csv(DATASET,index_col=0)
print('df read')

df_subset = df.copy()[subset_range[0]:subset_range[1]]
del df

df_subset['sentiment'] = df_subset['text'].progress_apply(sentiment_for_df)

filename = f'sentiment_subsample_{MODEL}_{DATASET}.csv'
df_subset.to_csv(f'{filename}')

print(f'{filename} saved')