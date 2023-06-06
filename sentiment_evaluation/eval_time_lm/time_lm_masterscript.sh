#!/bin/sh
FILES=(
    "2019_Q4"
    "2020_Q1"
    "2020_Q2"
    "2020_Q3"
    "2020_Q4"
    "2021_Q1"
    "2021_Q2"
    "2021_Q3"
    "2021_Q4"
    "2022_Q1"
    "2022_Q2"
    "2022_Q3"
    "2022_Q4"
)
for stamp in ${FILES[@]}; do

    FILE="submit_scripts/submit_${stamp}.sh"
    if [[ -f "$FILE" ]]; then
        echo $stamp
    else
        continue
    fi

    ###bsub -W 01:00 < "submit_scripts/submit_${stamp}.sh"
    bsub -W 01:00 < "submit_scripts/submit_${stamp}_pre.sh"
    bsub -W 01:00 < "submit_scripts/submit_${stamp}_post.sh"
done