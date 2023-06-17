# Transfer learning from pre-trained RoBERTa models

The intention with this part of the project is to improve pre-trained masked language models, from huggingface, with transfer learning.\
In this folder you will find four files:
- Quarterly_trainer, which is a .py script that takes a pre-trained model from huggingface as intput as well as a data set and returns a transfer learned model. OBS: The script uses wands and biases which might require a login and setup procedure.
- MLM_perplexity_eval, a .py script that takes a transfer learned model as input, as well as some validation data, and returns the performance measure of the model. In this case the performance is evaluated using perplexity (exponential of cross entropy loss).
- model
- data_pull_from_output