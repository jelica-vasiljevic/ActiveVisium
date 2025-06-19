export WANDB_DATA_DIR=""
export WANDB_API_KEY=""
export WANDB_BASE_URL=""
export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"

config_file=../config_files/experiments/test/practical/config_test.json
# dir where pathologist exports annotations
source_dir="../../ActiveVisium_practical_usecase/"
# dir where data is stored

target_dir=$(python bash_help/get_model_save_dir.py $config_file)

annotations='Pathologist_Annotations.csv'
prediction_pattern='Predictions_with_help_'



# if target directory does not exist, create it
mkdir -p $target_dir
cd ../src/practical_usage/
# copy the files at the end and beginning of the training
python3 Find_training_set.py  "../$config_file" "../$source_dir/$annotations" "$prediction_pattern"
exit_code=$?  # Capture the exit code

if [ "$exit_code" -eq 1 ]; then
    # Run the training script and subsequent steps only if Find_training_set.py exits with 1
    python3 ../01_Train.py "../$config_file" && \
    python3 ../02_Apply_model.py "../$config_file" && \
    python3 Add_help_to_predictions.py "../$config_file" "$prediction_pattern"
else
    echo "It seems that training data is not properly prepared. Please check did you remove all help spots or did you save file under name $annotations."
    exit 1
fi

cd ..
# copy all prediction files to target directory
cp "${target_dir}/${prediction_pattern}"* "${source_dir}"
cp "${target_dir}/ActiveVisium_annotations.csv" "${source_dir}"
echo "Training completed."

