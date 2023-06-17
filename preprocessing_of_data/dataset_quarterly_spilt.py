import os
import sys
import pandas as pd
from tqdm import tqdm

DATASET = sys.argv[1] # the is the endpath for the data set e.g. "validation_split.csv"
try:
    TARGET_FOLDER = sys.argv[2]
except IndexError:
    TARGET_FOLDER = "preprocessing/quarterly_splits"

YEARS = [2018, 2019, 2020, 2021, 2022]


try:
    data = pd.read_csv(f"{DATASET}",index_col=0,dtype={'text':str})
except:
    raise NameError(f'{DATASET} not found')

if not os.path.isdir(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)


if "date" not in list(data.columns):
    raise ValueError("No date column, please name this 'date' :)")

data['date'] = pd.to_datetime(data.date)
data['year'] = data.date.dt.year
data['quarter'] = data.date.dt.quarter


for year in YEARS:
    print(f'\nYear: {year}')
    for quarter in tqdm(range(1,5)):
        current_data = data[(data['year'] == year) & (data['quarter'] == quarter)].drop(['year','quarter'],axis=1)
        current_data.to_csv(f"{TARGET_FOLDER}/quarterly_data_{year}_Q{quarter}.csv")

