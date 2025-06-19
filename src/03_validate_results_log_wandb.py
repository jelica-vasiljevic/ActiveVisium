"""
Evaluate model predictions against ground truth annotations and log results to WandB.

Usage:
    python evaluate_predictions.py config.json [--use_wandb]

Arguments:
    config.json          Path to the configuration file.
    --use_wandb         Enable Weights & Biases logging (optional).
"""


import argparse
import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score
import os
import warnings
import wandb
from utils import config_utils
from global_constants import gt_column_name, colname_in_pred, max_num_active_learning_runs
# suppress warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions for ActiveVisium.")
    parser.add_argument("config_file_path", help="Path to configuration file.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging.")
    # defult should be True as this script is intended to log results to WandB
    parser.set_defaults(use_wandb=True)
    args = parser.parse_args()

    # Load user-provided configuration file
    my_configs = config_utils.read_config_from_user_file(args.config_file_path)
    test_case = my_configs["test_case_name"]

    pathologist_annotations =  my_configs['original_annotations_test_only']
    # Directory where model outputs and configs are saved
    runs_home_dir = os.path.join(os.path.dirname(my_configs['model_savepath']))

    #  Reload config from run directory if it exists (ensures latest config)
    config_file_path = os.path.join(runs_home_dir,'config.json')
    if os.path.exists(config_file_path):
        my_configs = config_utils.read_config_from_file(config_file_path)
    else:
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}, there is likely no run to validate.")
    
    # Stop if maximum number of active learning runs reached
    if my_configs['num_active_learning_runs']>=max_num_active_learning_runs:
        print("Reached maximum number of active learning runs")
        os._exit(0)

    # Initialize WandB run if requested
    if args.use_wandb and 'wandb_run_id' in my_configs:
        run_id = my_configs['wandb_run_id']
        run  = wandb.init(id=run_id,
                        resume="allow",
                        project=my_configs['project'])
    else:
        run = None
    
    # Find prediction files matching the test case, excluding training set predictions
    predicitons = [filename for filename in os.listdir(runs_home_dir) if filename.startswith('Model_predictions_'+test_case) and ('trainingset_predictions' not in filename)]
    # sort predicition by name in reverse order
    predicitons.sort(reverse=True)
    if not predicitons:
        raise FileNotFoundError(f"No prediction files found for test case {test_case} in {runs_home_dir}.")
    
    pred = predicitons[0]

    # Determine which test datasets are available for evaluation
    test_case_names = ['whole_image']
    if 'valid_data' in my_configs and os.path.exists(my_configs['valid_data']):
        test_case_names.append('valid_set')
        validation_data = my_configs['valid_data']
    if 'test_data' in my_configs and os.path.exists(my_configs['test_data']):
        test_case_names.append('test_set')
        test_data=  my_configs['test_data']

    # Evaluate each available test dataset
    for test_case in test_case_names:
        
        if test_case == 'whole_image':
            gt_data = pd.read_csv(pathologist_annotations)
        elif test_case == 'test_set':
            gt_data = pd.read_csv(test_data)
        elif test_case == 'valid_set':
            gt_data = pd.read_csv(validation_data)
        else:
            raise ValueError('test_case should be whole_image or test_set')
        
        predictions_data = pd.read_csv(os.path.join(runs_home_dir,pred))

       # Optionally handle missing spots in ground truth (currently disabled)
        take_into_account_missing_spots_in_gt = False
        if take_into_account_missing_spots_in_gt:
            merged_data = gt_data.merge(predictions_data, how='outer', on='Barcode')
            merged_data[gt_column_name] = merged_data[gt_column_name].fillna("missing_in_GT")
            merged_data[colname_in_pred] = merged_data[colname_in_pred].fillna("missing_in_pred")

        else:
            gt_data = gt_data[~gt_data[gt_column_name].isna()]
            merged_data = pd.merge(gt_data, predictions_data, on='Barcode')



        true_labels = merged_data[gt_column_name]
        predicted_labels = merged_data[colname_in_pred]

         # Compute balanced accuracy score
        blc_acc_score = balanced_accuracy_score(true_labels.to_list(), predicted_labels.to_list())
        # Log balanced accuracy score and classification report to WandB
        if run is not None:
            run.summary[test_case+"_overall_BAS"] = blc_acc_score  

        labels = list(set(merged_data[gt_column_name].unique()).union(merged_data[colname_in_pred].unique()))
        cls_report = classification_report(true_labels, predicted_labels, output_dict =True)
        
        if run is not None:
            # save in run summary f-score for each class
            for cls in labels:
                run.summary[f'f_score_{cls}'] = cls_report[cls]['f1-score']

            run.summary[test_case+"_overall_f_score"] = cls_report['macro avg']['f1-score']
            run.summary[test_case+"_weighted avg"] = cls_report['weighted avg']['f1-score']
        else:
            print(f"Balanced accuracy score for test case:{test_case} is {blc_acc_score}")
            print(f"Classification Report for test case:{test_case}")
            print(classification_report(true_labels, predicted_labels))



if __name__ == "__main__":
    
    if "WANDB_API_KEY" not in os.environ:
        print("WANDB_API_KEY not found in environment variables. Please set it before running the script.")
        use_wandb = False
    else:
        main()







