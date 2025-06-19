#!/bin/bash

source activate torch-os39_uni_model

# export WANDB_API_KEY=""
# export WANDB_BASE_URL=""
# export HUGGINGFACE_TOKEN=""

# configfilename=../config_files/experiments/test/unimodal/config_test.json
configfilename=../config_files/experiments/test/multimodal/config_test.json
# swict dir to src
cd ../src/

# extract patches
# echo "Extracting patches from the test image"
# cd data;
# python PatchExtractor.py -img ../../../data/test_data/CytAssist_FFPE_Human_Breast_Cancer_tissue_image_openslide.tif ;
# cd ..
# echo "Patches extracted successfully"
# # extract features
# echo "Extracting features from the patches"
# cd models;
# python extract_features.py ../${configfilename};
# cd ..
# echo "Features extracted successfully"

# Make inital set of annotations
cd dataset_split_first_time_exp
echo "Creating initial set of annotations"
python3 01_Inital_clustering_and_selection.py ../${configfilename}
echo "Simating annotations for the initial set"
python3 02_Simulate_annotations_first_time.py ../${configfilename}
echo "Initial set of annotations created successfully"
cd ..

echo "Starting the active learning loop for 10 iterations"

for i in {1..10}
do
    python3 01_Train.py ${configfilename}
    pid=$!
    echo "Waiting for 01_Train.py to finish, pid: ${pid}"
    wait $pid
    python3 02_Apply_model.py ${configfilename} 
    pid=$!
    echo "Waiting for 02_Apply_model.py to finish, pid: ${pid}"
    wait $pid
    python3 03_validate_results_log_wandb.py ${configfilename}
    pid=$!
    echo "Waiting for 03_validate_results_log_wandb.py to finish, pid: ${pid}"
    wait $pid
    cd experiments
    python3 04_Simulate_annotation.py ../${configfilename} 
    cd ..
    pid=$!
    echo "Waiting for 04_Simulate_annotation.py to finish, pid: ${pid}"
    wait $pid

done

echo "Active learning loop completed successfully"




