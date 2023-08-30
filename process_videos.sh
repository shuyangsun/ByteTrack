#!/bin/bash

# Usage: bash process_videos.sh /path/to/directory /path/to/output 0.25 0.0 cuda:0 # 0.25 is the fraction of files to process, 0.0 is the starting position fraction.

# Set the directory path, starting position ratio, and fraction of files to process
PATH_TO_BTRACK_MODEL="/home/ssun/developer/mlmodel/bytetrack/bytetrack_ablation.pth.tar"
input_dir=$1
output_dir=$2
fraction=$3 # e.g., 0.25 (4 batches)
starting_position_ratio=$4 # e.g., 0.75 (last batch)
device=$5

# Create an associative array to keep track of matched files from the second directory
declare -A matched_files

# Iterate through files in the second directory and populate the associative array
for second_file_path in "$output_dir"/*; do
    if [[ -f "$second_file_path" ]]; then
        second_file=$(basename "$second_file_path")
        file_name="${second_file%_result.txt}"
        matched_files["$file_name"]=1
    fi
done

no_match=()

# Iterate through files in the first directory and check for matches
for first_file_path in "$input_dir"/*; do
    if [[ -f "$first_file_path" ]]; then
        first_file=$(basename "$first_file_path")
        file_name="${first_file%.mp4}"
        
        if [[ ! ${matched_files["$file_name"]} ]]; then
            no_match+=("$input_dir/$first_file")
        fi
    fi
done

# Sort files
IFS=$'\n' file_list=($(sort <<<"${no_match[*]}"))
unset IFS

# Calculate the total number of files and the starting position based on ratios
total_files=${#file_list[@]}
start_position=$(echo "$total_files * $starting_position_ratio" | bc | awk '{print int($1)}')
num_files_to_process=$(echo "$total_files * $fraction" | bc | awk '{print int($1)}')
num_files_to_process_buf=$(echo "$num_files_to_process* 1.5" | bc | awk '{print int($1)}')

# Ensure the last batch processes all remaining files with 50% buffer.
if ((start_position + num_files_to_process_buf > total_files)); then
    num_files_to_process=$((total_files - start_position))
fi

echo "Processing ${num_files_to_process} video files..."

# Iterate through the selected files, starting from the calculated position, and process each one using the Python script
for ((i = start_position; i < start_position + num_files_to_process && i < total_files; i++)); do
    file_name="${file_list[$i]}"
    echo "Processing ${file_name}..."
    python tools/demo_track.py video -f exps/example/mot/yolox_x_ablation.py -c ${PATH_TO_BTRACK_MODEL} --device ${device} --fps 15 --fp16 --fuse --save_result --path ${file_name}
done
