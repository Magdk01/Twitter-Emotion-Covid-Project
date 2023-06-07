from transformers import pipeline
import pandas as pd
import tqdm
import os
import torch
import sys

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

DATASET = f"{sys.argv[1]}.csv"
MODEL = sys.argv[2] # See above for example

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'Using Cuda device: {device}\n')
else:    
    raise RuntimeError('No CUDA available')

tqdm.tqdm.pandas()

sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL,device=0,truncation=True,max_length=512)
print('pipeline loaded')

print('reading dataset...')
df = pd.read_csv(f'{DATASET}',index_col=0)
print('df read')

def sentiment_for_df(text):
	classification = sentiment_task(text)[0]['label']

	if classification =='negative':
		return -1
	elif classification =='neutral':
		return 0
	elif classification =='positive':
		return 1
	else:
		return classification
		
df['sentiment'] = df['text'].progress_apply(sentiment_for_df)

filename = f'./sentiment_subsample_{MODEL.split("/")[-1]}_{DATASET.split("/")[-1]}'
print(filename)
df.to_csv(f'{filename}')

print(f'{filename} saved')
