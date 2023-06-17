This is a step by step guide to doing the finetuning and the follow up analysis.

1.      Make sure u have the virtual enviroment setup with the correct packages seen in main README file

2.      Run the script Finetuning_with_Parametersearch.py, alternatively on the hpc run the script submit_pmsearch.sh

3.      Go into the newly created folder and find the parameters in Best_model_parameters.json, as these needs to be used in the next step

4.      Edit the file Finetuning_without_Parametersearch.py, to have the correct parameters, or the ones u want. Alternatively you can copy
        the trainer from Finetuning_with_Parametersearch.py and replace the current trainer if interested in training with a parametersearch
        in each TimeLM.

5.      Run the script Finetuning_without_Parametersearch.py, alternatively on the hpc run the script submit_finetuning.sh

6.      Inside the folder mckpt all models will be found, each folder for a model named "sentiment" and the number the model is. the folder
        called best_model is the model used for evaluation

7.      Make sure all best models are correctly named and in the correct folder (this being sentiment/base_model_#), or edit the file path in all the submit_20##_Q_#.sh 

8.      Run the script time_lm_masterscript.sh

9.      The results of each evaluation will be found in seperate csv files
