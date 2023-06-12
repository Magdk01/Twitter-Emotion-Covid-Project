# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:33:40 2023

@author: Mikkel
"""
import torch
import logging
import tweetnlp



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")
    
    
#logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# load dataset
dataset, label_to_id = tweetnlp.load_dataset("sentiment")
print("dataset loaded")
# load trainer class
trainer_class = tweetnlp.load_trainer("sentiment")
# setup trainer
print("trainer loaded")
trainer = trainer_class(
    language_model='cardiffnlp/twitter-roberta-base-2021-124m',  # language model to fine-tune
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_test='test',
    split_train='train',
    split_validation='validation',
    output_dir='model_ckpt/irony' 
)
print("class made")
# start model fine-tuning with parameter optimization
print(torch.cuda.is_available())
print(torch.cuda.current_device())
trainer.train(
  eval_step=50,  # each `eval_step`, models are validated on the validation set 
  n_trials=10,  # number of trial at parameter optimization
  search_range_lr=[1e-6, 1e-4],  # define the search space for learning rate (min and max value)
  search_range_epoch=[1, 6],  # define the search space for epoch (min and max value)
  search_list_batch=[4, 8, 16, 32, 64]  # define the search space for batch size (list of integer to test) 
)
# evaluate model on the test set
trainer.evaluate()
trainer.predict('If you wanna look like a badass, have drama on social media')
