import os
from glob import glob

models = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "cardiffnlp/twitter-roberta-base-sentiment",
]

time_lms = {
    "2019_Q4": "cardiffnlp/twitter-roberta-base-2019-90m",
    "2020_Q1": "cardiffnlp/twitter-roberta-base-mar2020",
    "2020_Q2": "cardiffnlp/twitter-roberta-base-jun2020",
    "2020_Q3": "cardiffnlp/twitter-roberta-base-sep2020",
    "2020_Q4": "cardiffnlp/twitter-roberta-base-dec2020",
    "2021_Q1": "cardiffnlp/twitter-roberta-base-mar2021",
    "2021_Q2": "cardiffnlp/twitter-roberta-base-jun2021",
    "2021_Q3": "cardiffnlp/twitter-roberta-base-sep2021",
    "2021_Q4": "cardiffnlp/twitter-roberta-base-dec2021",
    "2022_Q1": "cardiffnlp/twitter-roberta-base-mar2022",
    "2022_Q2": "cardiffnlp/twitter-roberta-base-jun2022",
    "2022_Q3": "cardiffnlp/twitter-roberta-base-sep2022",
    "2022_Q4": "cardiffnlp/twitter-roberta-base-2022-154m",
}

datafiles = glob("preprocessing/quarterly_splits/**.csv", recursive=True)
for file in datafiles:
    name = file.split("/")[-1].split(".")[0]
    stamp = "_".join(name.split("_")[-2], name.split("_")[-1])
    if stamp in time_lms.keys():
        os.system(f"python sentiment_eval_time_lm.py {name}.csv {time_lms[stamp]}")
        os.system(f"python sentiment_eval_time_lm.py {name}.csv {models[0]}")
        os.system(f"python sentiment_eval_time_lm.py {name}.csv {models[1]}")
        