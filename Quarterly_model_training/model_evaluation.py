from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd 
import numpy as np

from scipy.stats import chi2_contingency

def model_evaluation(dataset, model_path = False, tweeteval = "cardiffnlp/twitter-roberta-base-dec2020"):
    if not model_path:
        model_path = "./Quarterly_model_training/models/Q42019_model"
    tweeteval = "cardiffnlp/twitter-roberta-base-dec2020"

    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tweeteval)

    validation_set = list(pd.read_csv(dataset, header = 0, delimiter=",")['text'])

    encoded_inputs = tokenizer(validation_set, truncation=True, padding=True, return_tensors="pt", )
    labels = encoded_inputs.input_ids.clone()

    with torch.no_grad():
        outputs = model(**encoded_inputs, labels=labels)

    for i, sentence in enumerate(outputs.logits):
        correctness = np.sum([labels[i][j] == np.argmax(sentence[j]) for j in range(len(sentence))])
    
    # Calculate the loss
    # loss = outputs.loss

    return correctness

def chi_squared(v1, v2):
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
    return p_value

def compare_models(dataset, model_path1, model_path2):
    l1 = model_evaluation(dataset, model_path=model_path1)
    l2 = model_evaluation(dataset, model_path=model_path2)
    
    print(f"The mean of model1 is: {np.mean(l1)}, and model2: {np.mean(l2)}. \nWith a p_value of {chi_squared(l1, l2)}")
    return chi_squared(l1, l2)

if __name__ == "__main__":
    model_evaluation("./preprocessing/splits/validation_split.csv")