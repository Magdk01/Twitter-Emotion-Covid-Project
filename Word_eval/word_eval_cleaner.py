# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:21:57 2023

@author: Mikkel
"""
import os
import numpy as np
import pandas as pd
import re

name = 'sentiment_keywords'
df1 = pd.read_csv(name+'.csv',index_col=0)

filehelper = 0

for i in range(13):
    for j in range(47):
        
        long_string = df1.iloc[j,i+1]
        df1.iloc[j,i+1] = re.sub(r'[^0-9.]', '', long_string)
        numbers = np.array([])
        number = ''
        for l in range(len(df1.iloc[j,i+1])):
            if l > 2:
                if df1.iloc[j,i+1][l] == '.':
                    numbers = np.append(numbers,round(float(number),4))
                    number = ''
            number = number + df1.iloc[j,i+1][l]
            if l == len(df1.iloc[j,i+1])-1:
                    numbers = np.append(numbers,round(float(number),4))
            
        
        df1.iloc[j,i+1] = str(numbers[0])+' '+str(numbers[1])+' '+str(numbers[2])
filename = f'./proccesed_sentiment_keywords'
if os.path.isfile(filename) == True:
    a = True
    while a ==True:
        filehelper += 1
        if os.path.isfile(filename+str(filehelper)) == False:
            a = False
    

    
    

filename = f'./proccesed_sentiment_keywords'+str(filehelper)
print(filename)
df1.to_csv(f'{filename}')

print(f'{filename} saved')


                
                
            
            

