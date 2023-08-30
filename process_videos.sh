#!/bin/bash

# Usage: bash process_videos.sh /path/to/directory 0.25 0.0 # 0.25 is the fraction of files to process, 0.0 is the starting position fraction.

# Set the directory path, starting position ratio, and fraction of files to process
PATH_TO_BTRACK_MODEL="/home/ssun/developer/mlmodel/bytetrack/bytetrack_ablation.pth.tar"
directory_path=$1
fraction=$2                 # Change this to your desired fraction
starting_position_ratio=$3  # Change this to the desired starting position ratio

# Get the list of files in the directory
file_list=("$directory_path"*".mp4")

# Calculate the total number of files and the starting position based on ratios
total_files=${#file_list[@]}
start_position=$(echo "$total_files * $starting_position_ratio" | bc | awk '{print int($1)}')
num_files_to_process=$(echo "$total_files * $fraction" | bc | awk '{print int($1)}')

# Iterate through the selected files, starting from the calculated position, and process each one using the Python script
for ((i = start_position; i < start_position + num_files_to_process && i < total_files; i++)); do
    file_name="${file_list[$i]}"
    python tools/demo_track.py video -f exps/example/mot/yolox_x_ablation.py -c ${PATH_TO_BTRACK_MODEL} --fps 15 --fp16 --fuse --save_result --path ${file_name}
    python your_script.py "$file_name"
done
