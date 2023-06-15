# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:21:57 2023

@author: Mikkel
"""

import tweetnlp
import pandas as pd
import tqdm
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import sys
tqdm.tqdm.pandas()

DATASET = f"{sys.argv[1]}.csv"

model = ['base_model_0','base_model_1','base_model_2','base_model_3','base_model_4','base_model_5',
		'base_model_6','base_model_7','base_model_8','base_model_9','base_model_10'
		]
df = pd.read_csv(f'{DATASET}',index_col=0)

def wordstimelm(word):
		prediction = model.predict(word, return_probability = True)
		return prediction

for i,MODEL in enumerate(model):
	model = tweetnlp.load_model('sentiment',MODEL)		
	df['sentimenttime_lm'+str(i)] = df['keyword'].progress_apply(wordstimelm)


def sentiment_scoring(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    return scores

premodel = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
postmodel = f"cardiffnlp/twitter-roberta-base-sentiment"   

tokenizer = AutoTokenizer.from_pretrained(premodel)
config = AutoConfig.from_pretrained(premodel)
model = AutoModelForSequenceClassification.from_pretrained(premodel)

df['sentimentpre'] = df['keyword'].progress_apply(sentiment_scoring)

tokenizer = AutoTokenizer.from_pretrained(postmodel)
config = AutoConfig.from_pretrained(postmodel)
model = AutoModelForSequenceClassification.from_pretrained(postmodel)

df['sentimentpost'] = df['keyword'].progress_apply(sentiment_scoring)

filename = f'./sentiment_keywords'
print(filename)
df.to_csv(f'{filename}')

print(f'{filename} saved')





