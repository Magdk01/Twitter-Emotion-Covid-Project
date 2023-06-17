# Setting up and running preproccesing of datasets to use for huggingface models
A short manual for generating the required data structure to preprocess and split dataset to later use with HuggingFace models.

## 1. Raw data
To begin generation of datasets to use in huggingface models first a folder named "raw_data" must be made locally. 

In the raw_data folder .csv files with tweets and dates should be placed. 

## 2. Preprocess data
When data has been placed into the "raw_data" folder the notebook "data_preprocess.ipynb" can be run. Note that this has been set up for the specific format of our .csv files(columns: 'id', 'date', 'text') and you might have to format other files accordingly.

If the .csv files fit the format the entire notebook can be run and i will generate a folder called "processed_year_data" and fill it up with the processed raw data containing the needed columns: 'id', 'date', 'text' and with all duplicates removed as well as usernames and links being replaced by their respective tokens.

Lastly the notebook takes all generatated files and concatenates them into 1 .csv file and outputs it as "collected_processed.csv" in the same folder.

## 3. Splitting data
When the data has been preprocessed it can be split into "train", "test", and "validation" by running "dataset_spliter.ipynb", which will create a folder "splits" and insert the corresponding files.

## 4. Feature extractions
To generate the list of keywords of intrest from the original dataset, the script "feature_list_extraction.py" can be run and it will generate a "inital_keywords.csv" file in the "raw_data" folder.