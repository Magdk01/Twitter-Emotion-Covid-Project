from transformers import pipeline
import pandas as pd
import tqdm
import os
import torch
import sys


#Sets the cache folder for the transformer models on the HPC's HOME drive
os.environ['TRANSFORMERS_CACHE'] = './cache/'

if __name__ == '__main__':

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

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL,device=0,truncation=True,max_length=512)
    print('pipeline loaded')


    def sentiment_for_df(text):
    # print(text)
        classification = sentiment_task(text)[0]['label']
        # print(classification)
        if classification =='negative':
            return -1
        elif classification =='neutral':
            return 0
        elif classification =='positive':
            return 1

    print('reading dataset...')
    df = pd.read_csv("./pilot_split.csv",index_col=0)
    print('df read')
    # dataset = Dataset.from_pandas(df)
    
    df_subset = df.copy()[subset_range[0]:subset_range[1]]
    del df

    df_subset['sentiment'] = df_subset['text'].progress_apply(sentiment_for_df)

    filename = f'sentiment_subsample_{subset_range[0]}_{subset_range[1]}.csv'
    df_subset.to_csv(f'{filename}')

    print(f'{filename} saved')