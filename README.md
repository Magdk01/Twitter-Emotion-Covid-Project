# Twitter-Emotion-Covid-Project
Project work for the DTU project course:
"02466 Project work - Bachelor of Artificial Intelligence and Data Spring 23".

### Project title:
Twitter emotions  - Did emotions change during the pandemic when tweeting about nature?

### Description
This repo contains the full code to reproduce analysis done in abovementioned project rapport.\
The main idea is to take a big data set of tweets scraped using nature related keywords. This repo then contians functionality to pre-process data, use transfer learning to improve existing huggingface MLM's, fine tune MLM's to sentiment down stream tasks, and evaluate sentiment using fine tuned models, either pre trained or transfer learned and fintuned.\
In general the repo is build with six scripts, most of them containing a script to perform an analysis and a script to evaluate the analysis afterwords. For each folder is one type of analysis, which are all described using local README files.

### Step-by-step guide
To run the entire analysis, use this script (it is not recommended to do this on a local computer, use an HPC)

1. Place raw data in the "preprocessing of data" folder (data can be in several different files if its to big to fit in one). Then follow the README in that folder to preprocess data.
2. Start transfer learning. Do this using the scripts (and explanatoriy README) in transfer learning folder. This might take a while.
3. Run the fine tuning scripts in the "finetuning to sentiment analysis" folder. This again might take a while.
3. Now its time for sentiment evaluation. This can be done earlier, as this process doesn't rely on any of the other processes (except the preprocessing).
4. Lastly use the statics script to evaluate on relevant methods.