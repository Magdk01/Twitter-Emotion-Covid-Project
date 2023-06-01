import os
import sys
import pandas as pd

DATASET = sys.argv[1] # the is the endpath for the data set e.g. "validation_split.csv"
try:
    TARGET_FOLDER = sys.argv[2]
except IndexError:
    TARGET_FOLDER = "preprocessing/quarterly_splits"

YEARS = [2018, 2019, 2020, 2021, 2022]

if not (cwd:=os.getcwd().split('\\')[-1]) == 'Twitter-Emotion-Covid-Project':
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
    assert os.getcwd().split('\\')[-1] == 'Twitter-Emotion-Covid-Project', 'Working directory is wrong'
print(f'Parent folder is set to: {os.getcwd()}')

if not os.path.isdir(TARGET_FOLDER):
    os.mkdir(TARGET_FOLDER)

data = pd.read_csv(f"preprocessing/splits/{DATASET}")

if "date" not in list(data.columns):
    raise ValueError("No date column, please name this 'date' :)")

data['date'] = pd.to_datetime(data.date)
data['year'] = data.date.dt.year
data['quarter'] = data.date.dt.quarter

for year in YEARS:
    for quarter in range(1,5):
        current_data = data[(data['year'] == year) & (data['quarter'] == quarter)]
        current_data.to_csv(f"{TARGET_FOLDER}/quarterly_data_{year}_Q{quarter}.csv")


