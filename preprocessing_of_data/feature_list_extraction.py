import pandas as pd
import numpy as np
import tqdm
import glob
#Extraction of the unqiue features from the raw data feature extraction
feature_list = []
for file in tqdm.tqdm(glob.glob('raw_data/CES_*.csv')):
    feature_list.append( pd.unique(pd.read_csv(file).feature))
pd.DataFrame((list(pd.unique(np.concatenate(feature_list)))),columns=['keyword']).to_csv('raw_data/initial_keywords.csv')