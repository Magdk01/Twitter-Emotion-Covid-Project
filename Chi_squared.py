# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:06:26 2023

@author: Mikkel
"""

import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import pandas as pd

filespre = ['sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2018_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2018_Q2','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2018_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2018_Q4','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2019_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2019_Q2','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2019_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2019_Q4','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2020_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2020_Q2','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2020_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2020_Q4','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2021_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2021_Q2','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2021_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2021_Q4','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2022_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2022_Q2','sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2022_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment_quarterly_data_2022_Q4'
         ]


filespost = ['sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2018_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2018_Q2','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2018_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2018_Q4','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2019_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2019_Q2','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2019_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2019_Q4','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2020_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2020_Q2','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2020_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2020_Q4','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2021_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2021_Q2','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2021_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2021_Q4','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2022_Q1',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2022_Q2','sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2022_Q3',
         'sentiment_subsample_twitter-roberta-base-sentiment-latest_quarterly_data_2022_Q4'
         ]

filestimelm = ['sentiment_subsample_base_model_0_quarterly_data_2020_Q1',
         'sentiment_subsample_base_model_1_quarterly_data_2020_Q2','sentiment_subsample_base_model_2_quarterly_data_2020_Q3',
         'sentiment_subsample_base_model_3_quarterly_data_2020_Q4','sentiment_subsample_base_model_4_quarterly_data_2021_Q1',
         'sentiment_subsample_base_model_5_quarterly_data_2021_Q2','sentiment_subsample_base_model_6_quarterly_data_2021_Q3',
         'sentiment_subsample_base_model_7_quarterly_data_2021_Q4','sentiment_subsample_base_model_8_quarterly_data_2022_Q1',
         'sentiment_subsample_base_model_9_quarterly_data_2022_Q2','sentiment_subsample_base_model_10_quarterly_data_2022_Q3'
         ]
cprepost =  []
cpretimelm =  []
cpostimelm =  []
pprepost = []
ppretimelm = []
pposttimelm = []
c = []
p = []

for j in range(3):
    if j == 0:
        filesm1 = filespre
        filesm2 = filespost
    elif j == 1:
        for g in range(8):
            filespre.remove(filespre[0])
            filespost.remove(filespost[0])
        filespre.remove(filespre[len(filespre)-1])
        filespost.remove(filespost[len(filespost)-1])
        cprepost = c
        c = []
        pprepost = p
        p = []
        filesm1 = filestimelm
        filesm2 = filespre
    else:
        cpretimelm = c
        c = []
        ppretimelm = p
        p = []
        filesm1 = filestimelm
        filesm2 = filespost
   #loop that does the chi-squared test
    for i,name in enumerate(filesm1):
        
        # load data of model 1
        df1 = pd.read_csv(name+'.csv')
        
        # load data of model 2
        df2 = pd.read_csv(filesm2[i]+'.csv')
        
        #put into vector
        #due to all the premodels having LABEL_0, LABEL_1 or LABEL_2 instead of -1, 0 or 1
        #we had to change these to integers, they have to be integers to fit into chi-squared later
        v1 = df1.iloc[:, -1].values
        if v1[0] != -1 and v1[0] !=0 and v1[0] !=1 and ('roberta' in name):
            b = 0
            for v in v1:
                if v == 'LABEL_0':
                    v1[b] = int(-1)
                elif v == 'LABEL_1':
                    v1[b] = int(0)
                else:
                    v1[b] = int(1)
                b += 1
        
        #issue in timelm not being the three integers but instead a dictionary.
        if v1[0] != -1 and v1[0] !=0 and v1[0] !=1 and ('model' in name):
            b = 0
            for v in v1:
                if 'negative' in v:
                    v1[b] = int(-1)
                elif 'neutral' in v:
                    v1[b] = int(0)
                elif 'positive'in v:
                    v1[b] = int(1)
                b += 1
        
        #put into vector
        v2 = df2.iloc[:, -1].values
        #same issue as earlier with premodels
        if v2[0] != -1 and v2[0] !=0 and v2[0] !=1:
            b = 0
            for v in v2:
                if v == 'LABEL_0':
                    v2[b] = int(-1)
                elif v == 'LABEL_1':
                    v2[b] = int(0)
                elif v == 'LABEL_2':
                    v2[b] = int(1)
                b += 1
        
        # chisquared test
                
        #this is needed cause v2 last value when it is postmodel is nan
        v2 = np.delete(v2, len(v2)-1)
        
        contingency_table = np.zeros((2, 3), dtype=int)
        for a in range(len(v1)+len(v2)):
            
            if a < len(v1):
                row = 0
                col = v1[a]+1
            else:
                row = 1
                col = v2[a-len(v1)]+1
            row = int(row)
            col = int(col)
            contingency_table[row][col] += 1
            
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        c.append(chi2)
        p.append(p_value)
cposttimelm = c
pposttimelm = p
