#!/bin/sh
models=(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
    "cardiffnlp/twitter-roberta-base-sentiment"
)

declare -A time_lms
time_lms=(
    ["2019_Q4"]="cardiffnlp/twitter-roberta-base-2019-90m"
    ["2020_Q1"]="cardiffnlp/twitter-roberta-base-mar2020"
    ["2020_Q2"]="cardiffnlp/twitter-roberta-base-jun2020"
    ["2020_Q3"]="cardiffnlp/twitter-roberta-base-sep2020"
    ["2020_Q4"]="cardiffnlp/twitter-roberta-base-dec2020"
    ["2021_Q1"]="cardiffnlp/twitter-roberta-base-mar2021"
    ["2021_Q2"]="cardiffnlp/twitter-roberta-base-jun2021"
    ["2021_Q3"]="cardiffnlp/twitter-roberta-base-sep2021"
    ["2021_Q4"]="cardiffnlp/twitter-roberta-base-dec2021"
    ["2022_Q1"]="cardiffnlp/twitter-roberta-base-mar2022"
    ["2022_Q2"]="cardiffnlp/twitter-roberta-base-jun2022"
    ["2022_Q3"]="cardiffnlp/twitter-roberta-base-sep2022"
    ["2022_Q4"]="cardiffnlp/twitter-roberta-base-2022-154m"
)
for file in *.csv; do
    IFS='.' read -ra name <<< "$file"
    name=${name[0]}
    IFS='_' read -ra stamp <<< "$name"
    len=${#stamp[@]}
    if [ $len -lt 2 ]; then
        echo "File path is not working: {$name}"
        continue
    fi
    stamp="${stamp[len-2]}_${stamp[len-1]}"
    name="$name.csv"
    
    echo $name
    echo "${time_lms[$stamp]}"

    bash ./submit_time_lm.sh $name "${time_lms[$stamp]}"
done