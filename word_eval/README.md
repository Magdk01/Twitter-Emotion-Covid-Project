# Zero shot evaluation

This folder contains the files to do a zero shot evaluation of a model.
It contains a "sumbit_word.sh" script to control the hyperparameters fro the HPC and submit relevant scripts. 
A "Timelm_word_eval.py" script that takes a series of models and evaluates the sentiment of a series of words found in the evaluation_keywords.csv file. Model have to be trained using the hugging face api and saved using model.save()
The last script formats the output of the former script and puts the formattet version in a csv file ("proccesed_sentiment_keywords)