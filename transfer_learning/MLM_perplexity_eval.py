import torch
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling
from tqdm import tqdm
import numpy as np
from scipy import stats
import sys
import re
# Define tokenizer and model

torch.maunal_seed(42)
np.random.seed(42)

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


stamp = sys.argv[1]
model = time_lms[stamp]

tokenizer = RobertaTokenizerFast.from_pretrained(model)
model1 = RobertaForMaskedLM.from_pretrained(model)
model2 = RobertaForMaskedLM.from_pretrained(f"Train_{stamp}",local_files_only = True)

# Move models to GPU if CUDA is available
device = torch.device("cuda:0")
model1 = model1.to(device)
model2 = model2.to(device)

# Load the dataframe
df_main = pd.read_csv(f"data/quarterly_data_{stamp}.csv",usecols=['text'])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
perplexity_list1 = []
perplexity_list2 = []
loss_fct = torch.nn.CrossEntropyLoss()

def model_perplex(model, inputs):
    model = model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate Cross Entropy Loss
    
    masked_lm_loss = loss_fct(outputs.logits.view(-1, model.config.vocab_size), inputs["labels"].view(-1))

    # Compute perplexity
    perplexity = torch.exp(masked_lm_loss).item()

    return perplexity


batch_size = 100
iteration_size = int(np.floor(len(df_main)/batch_size))

for i in tqdm(range(iteration_size)):
    df = df_main[i*batch_size:batch_size+i*batch_size]
    sentences = list(df['text'])

    # Prepare the data and mask tokens
    inputs = tokenizer(sentences, truncation=True, padding=True)
    inputs = data_collator(inputs['input_ids'])

    # Move inputs to GPU
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Model prediction
    perplexity_list1.append(model_perplex(model1, inputs))
    perplexity_list2.append(model_perplex(model2, inputs))

# Assuming you have two arrays of perplexity scores
perplexities_model1 = np.array(perplexity_list1)  
perplexities_model2 = np.array(perplexity_list2)  

# Calculate the paired t-test
t_statistic, p_value = stats.ttest_rel(perplexities_model1, perplexities_model2)

print("t statistic:", t_statistic)
print("p value:", p_value)

print(f'Average perplexity of base model: {np.mean(perplexities_model1)}')
print(f'Average perplexity of transfer-learned model: {np.mean(perplexities_model2)}')
