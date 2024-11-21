#!/bin/bash


EXPERIMENT_NAME=Folder1_Folder2
FOLDER=.data/Folder1/Folder2
export CUDA_VISIBLE_DEVICES=1

# Check if the folder exists; if not, create it
mkdir -p ./Results/"$EXPERIMENT_NAME"

rm ./Results/"$EXPERIMENT_NAME"/*.png
rm ./Results/"$EXPERIMENT_NAME"/"$EXPERIMENT_NAME".log
rm ./Results/"$EXPERIMENT_NAME"/"$EXPERIMENT_NAME".csv

nohup apptainer exec --nv ./container_vphys.sif python ./main.py --dt 3/60 --path "$FOLDER"  --outfolder "$EXPERIMENT_NAME"  >> ./Results/"$EXPERIMENT_NAME"/"$EXPERIMENT_NAME".log 2>&1 &


