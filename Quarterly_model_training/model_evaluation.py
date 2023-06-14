from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd 
import numpy as np
from tqdm import tqdm
from glob import glob
import sys
from datetime import datetime

from scipy.stats import chi2_contingency

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

def model_evaluation(dataset, model_path = False, tweeteval = "cardiffnlp/twitter-roberta-base-dec2020"):
    
	model = AutoModelForMaskedLM.from_pretrained(model_path)
	tokenizer = AutoTokenizer.from_pretrained(tweeteval)

	data_files = glob("data/*.csv")
	correctness = list()
	for jj, file in enumerate(data_files):
		validation_set = list(pd.read_csv(file, header = 0, delimiter=",", usecols=["text"])["text"])

		encoded_inputs = tokenizer(validation_set, truncation=True, padding=True, return_tensors="pt", )
		labels = encoded_inputs.input_ids.clone()

		with torch.no_grad():
			outputs = model(**encoded_inputs, labels=labels)

		for i, sentence in enumerate(outputs.logits):
			for j in range(len(sentence)):
				correctness.append(1 if labels[i][j] == np.argmax(sentence[j]) else 0)
		print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}, Itteration count: {jj}")
	return np.array(correctness)

def chi_squared(v1, v2):
	contingency_table = np.zeros((2, 2), dtype=int)
	for i in range(len(v1)):
		contingency_table[0][v1[i]] += 1
		contingency_table[1][v2[i]] += 1
		
	chi2, p_value, dof, expected = chi2_contingency(contingency_table)
	return p_value

def compare_models(dataset, model_path1, model_path2):
	l1 = model_evaluation(dataset, model_path=model_path1)
	l2 = model_evaluation(dataset, model_path=model_path2)
	print(f"v1: {l1}\nv2: {l2}")
    
	print(f"The mean of model1 is: {np.mean(l1)}, and model2: {np.mean(l2)}. \nWith a p_value of {chi_squared(l1, l2)}")
	return chi_squared(l1, l2)

if __name__ == "__main__":
	stamp = sys.argv[1]
	compare_models("data/validation_split.csv", f"./Train_{stamp}", time_lms[stamp])
