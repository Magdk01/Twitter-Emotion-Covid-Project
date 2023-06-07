# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:06:26 2023

@author: Mikkel
"""

import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import pandas as pd

# load data of model 1
df1 = pd.read_csv('merged_subsamples.csv')

# load data of model 2
#df2 =pd.read_csv('filename')

#put into vector
v1 = df1.iloc[:, -1].values

#put into vector
#v2= df2.iloc[:, -1].values



#Vectors, fix to be vectors containing our sentiment scores from different models
np.random.seed(42)
#v1 = np.random.choice([-1, 0, 1], size=30000000)
v2 = np.random.choice([-1, 0, 1], size=len(v1))


# Mann-Whitney U test
statistic, p_value = mannwhitneyu(v1, v2)

print(f"Mann-Whitney U statistic: {statistic}")
print(f"P-value: {p_value}")

# chisquared test

contingency_table = np.zeros((2, 3), dtype=int)

for i in range(len(v1)+len(v2)):
    
    if i < len(v1):
        row = 0
        col = v1[i]
    else:
        row = 1
        col = v2[i-len(v1)]
    contingency_table[row][col] += 1

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")