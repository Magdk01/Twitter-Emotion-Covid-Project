# Fine tuning to sentiment downstream task

Code in this folder is meant to take a pre-trained RoBERTa model from the huggingface api as input and output a fine tuned versin of the same model. \
As the title suggests the fine tuning in this code is for sentiment analysis. \
\
You will find three folders: \
- finetuning_with_parametersearch, which contains a .py script that performs a fine tuning of a given model and searches for optimal hyperparameters and a .sh script to submit this to an HPC. This might take a while.
- finetuning_without_parametersearch, wich contains the same as above, only this version does not search for hyperparameters, these have to be inputtet. This saves a lot of time a can be used for faster results.
- evaluation_sentiment_timelm, which containes a folder for .sh scripts that can be called using the time_lm_maserscript.sh to submit the .py scripts to the HPC with different parameters. This is meant to evaluate the sentiment from different models on different data sets.

If this is used to fine tune several alike models it can be smart to only fine tune one with hyperparameter search and the utilize those hyperparameters for the other models.