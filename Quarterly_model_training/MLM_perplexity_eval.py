import torch
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling
from tqdm import tqdm
import numpy as np
from scipy import stats
import sys
import re
# Define tokenizer and model

model, year_quarter= sys.argv[1], sys.argv[2]
assert re.search("\d{4}_Q\d",year_quarter).group(0), 'Doesnt recognize'

tokenizer = RobertaTokenizerFast.from_pretrained(f"cardiffnlp/{model}")
model1 = RobertaForMaskedLM.from_pretrained(f"cardiffnlp/{model}")
model2 = RobertaForMaskedLM.from_pretrained('Train_2021_Q3',local_files_only = True)

# Move models to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model2 = model2.to(device)

# Load the dataframe
df_main = pd.read_csv(f"validation_splits/{year_quarter}_validation_text_only.csv",index_col=0)
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