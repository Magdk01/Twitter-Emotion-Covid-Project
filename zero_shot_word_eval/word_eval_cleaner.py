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

#removes the part of the string that contained "LABEL:" and a number, the number persisted in the
#removal of characters that arent numbers or "."
for i in range(47):
    df1.iloc[i,13] = df1.iloc[i,13][46:]

for i in range(13):
    for j in range(47):
        #deletes all characters in the array that are not a number or "."
        df1.iloc[j,i+1] = re.sub(r'[^0-9.]', '', df1.iloc[j,i+1])
        numbers = np.array([])
        number = ''
        #separates numbers, make them float such that they can be rounded, and replaces teh string in df1
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
#makes sure you do not overwrite ur own file
if os.path.isfile('proccesed_sentiment_keywords'+str(filehelper)+'.csv') == True:
    a = True
    while a ==True:
        filehelper += 1
        if os.path.isfile(filename+str(filehelper)+'.csv') == False:
            a = False
    

    
    

filename = f'./proccesed_sentiment_keywords'+str(filehelper)
print(filename)
df1.to_csv(f'{filename}')

print(f'{filename} saved')


                
                
            
            

