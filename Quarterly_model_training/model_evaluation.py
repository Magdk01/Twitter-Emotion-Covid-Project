from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pandas as pd 

def fix_dataset(df):
    df = df.replace(r'\r+|\t+','', regex=True)
    df = df.replace(r'\n',' ', regex=True)
    return df    

def model_evaluation(dataset, model_path = False, tweeteval = "cardiffnlp/twitter-roberta-base-dec2020"):
    if not model_path:
        model_path = "./Quarterly_model_training/models/Q42019_model"
    tweeteval = "cardiffnlp/twitter-roberta-base-dec2020"

    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tweeteval)

    validation_set = list(fix_dataset(pd.read_csv(dataset, header = 0, delimiter=","))['text'])[:10]

    encoded_inputs = tokenizer(validation_set, truncation=True, padding=True, return_tensors="pt")
    labels = encoded_inputs.input_ids.clone()

    with torch.no_grad():
        outputs = model(**encoded_inputs, labels=labels)

    # Calculate the loss
    loss = outputs.loss

    return loss.item()