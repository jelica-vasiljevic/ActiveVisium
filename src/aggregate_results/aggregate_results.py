"""
This script evaluates model predictions across multiple experiments and active learning iterations.

It aggregates results from various configurations, computes performance metrics 
(F1-score, weighted F1, balanced accuracy, confusion matrix), and stores the results 
in CSV files for downstream analysis and visualization.
"""


import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from utils import config_utils
from global_constants import *

warnings.filterwarnings("ignore")


# ------------------------ Configuration ------------------------

DATASET = 'mouse_kidney'
DATASET_TO_VALIDATE = 'test'  # Options: 'annotation', 'annotation&test', 'test', 'transfer_learning'
CONFIG_FILE_PATH = f'../../config_files/experiments/{DATASET}/fully_supervised/' # unimodal, multimodal, fully_supervised
FULL_DATASET_SIZE=-1

if "fully_supervised" in CONFIG_FILE_PATH:
    FULLY_SUPERVISED = True
else:
    FULLY_SUPERVISED = False


def extract_base_name(config_file, fully_supervised=False):
    filename = os.path.basename(config_file)
    if not fully_supervised:
        pattern = r'config_(.*?)_(.*?)_(.*?)(_ICF_\d+)?\.json'
        match = re.match(pattern, filename)
        if match:
            return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    else:
        pattern = r'config_(.*?)(_(.*?)){0,1}_[0,1,2]{1}.json'
        match = re.match(pattern, filename)
        if match:
            second_part = match.group(2) or ""
            return f"{match.group(1)}{second_part}"
    return None

def list_config_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]

def load_ground_truth(config, dataset_mode):
    if dataset_mode == 'transfer_learning':
        gt =  pd.read_csv(config['original_annotations_test_only'])
    elif dataset_mode == 'test':
        gt =  pd.read_csv(config['test_data'])
    elif dataset_mode == 'annotation':
        original = pd.read_csv(config['original_annotations_test_only'])
        test = pd.read_csv(config['test_data'])
        gt =  original[~original['Barcode'].isin(test['Barcode'])]
    elif dataset_mode == 'annotation&test':
        gt =  pd.read_csv(config['original_annotations_test_only'])
    else:
        raise ValueError("Invalid dataset_to_validate option.")
    
    if FULLY_SUPERVISED:
        global FULL_DATASET_SIZE
        FULL_DATASET_SIZE = gt.shape[0]

    return gt

def evaluate_config(config_file, gt_data, test_case, fully_supervised):
    config = config_utils.read_config_from_user_file(config_file)
    runs_dir = os.path.dirname(config['model_savepath'])
    print("\033[91mConfig file:\033[0m", config_file)
    print("\033[91mRuns directory:\033[0m", runs_dir)
    if fully_supervised:
        prediction_files = [
            f for f in os.listdir(runs_dir)
            if f.startswith(f'Model_predictions_{test_case}') and 'no_trainingset_predictions_corrected' in f
        ]
    else:
        prediction_files = [
            f for f in os.listdir(runs_dir)
            if f.startswith(f'Model_predictions_{test_case}') and 'no_trainingset_predictions_corrected' not in f
        ]

    if len(prediction_files) != 10 and not fully_supervised:
        raise ValueError(f'Expected 10 predictions, found {len(prediction_files)}')

    prediction_files.sort(reverse=True)

    results = {
        'f_scores': [], 'weighted_f_scores': [], 'balanced_accuracies': [],
        'misclassified_samples': [], 'misclassified_per_class': [],
        'reports': [], 'annotated_counts': [], 'annotated_per_class': []
    }

    for file in prediction_files:
        pred_data = pd.read_csv(os.path.join(runs_dir, file))
        merged = pd.merge(gt_data, pred_data, on='Barcode')
        y_true, y_pred = merged[gt_column_name], merged[colname_in_pred]

        report = classification_report(y_true, y_pred, output_dict=True)
        f1, wf1 = report["macro avg"]["f1-score"], report["weighted avg"]["f1-score"]
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Processing {file}\nBalanced Accuracy: {bal_acc:.4f}\nConfusion Matrix:\n{cm}")

        results['f_scores'].append(f1)
        results['weighted_f_scores'].append(wf1)
        results['balanced_accuracies'].append(bal_acc)
        results['reports'].append(report)

        misclassified = merged[y_true != y_pred]
        results['misclassified_samples'].append(len(misclassified))
        results['misclassified_per_class'].append(misclassified[gt_column_name].value_counts().to_dict())

    # Annotations
    train_files = sorted(
        [f for f in os.listdir(runs_dir) if f.startswith('TrainingAnnotations_')],
        reverse=True
    )

    for file in train_files:
        df = pd.read_csv(os.path.join(runs_dir, file))
        df = df[df[gt_column_name] != 'help']
        results['annotated_counts'].append(len(df))
        results['annotated_per_class'].append(df[gt_column_name].value_counts().to_dict())

    return results, prediction_files

def save_results(all_results, pred_files, output_dir, dataset_mode):
    results_summary = []
    detailed_report = []

    for config_file, results in all_results.items():
        base_name = extract_base_name(config_file, fully_supervised=FULLY_SUPERVISED)
        for i in range(len(pred_files)):
            results_summary.append({
                'Config File': config_file,
                'Base_name': base_name,
                'Iteration': len(pred_files) - 1 - i,
                'F-score': results['f_scores'][i],
                'weighted F-score': results['weighted_f_scores'][i],
                'Balanced Accuracy': results['balanced_accuracies'][i],
                'Misclassified Samples': results['misclassified_samples'][i],
                'Total Annotated Data': results['annotated_counts'][i] if not FULLY_SUPERVISED else FULL_DATASET_SIZE,
            })

            for class_label in results['reports'][i].keys():
                if class_label in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                detailed_report.append({
                    'Config File': config_file,
                    'Base_name': base_name,
                    'Iteration': len(pred_files) - 1 - i,
                    'Class Label': class_label,
                    'F-score': results['reports'][i][class_label]['f1-score'],
                    'Precision': results['reports'][i][class_label]['precision'],
                    'Recall': results['reports'][i][class_label]['recall'],
                    'Support': results['reports'][i][class_label]['support'],
                    'Annotations_provided': results['annotated_per_class'][i].get(class_label, 0) if not FULLY_SUPERVISED else FULL_DATASET_SIZE,
                    'Misclassified Samples': results['misclassified_per_class'][i].get(class_label, 0)
                })

    pd.DataFrame(results_summary).to_csv(f"{output_dir}/results_{dataset_mode}.csv", index=False)
    pd.DataFrame(detailed_report).to_csv(f"{output_dir}/classification_reports_{dataset_mode}.csv", index=False)

if __name__ == "__main__":
    config_files = list_config_files(CONFIG_FILE_PATH)
    base_config = config_utils.read_config_from_user_file(config_files[0])
    test_case_name = base_config["test_case_name"]
    ground_truth = load_ground_truth(base_config, DATASET_TO_VALIDATE)

    all_results = {}
    for file in config_files:
        all_results[file], prediction_files = evaluate_config(file, ground_truth, test_case_name, FULLY_SUPERVISED)

    save_results(all_results, prediction_files, CONFIG_FILE_PATH, DATASET_TO_VALIDATE)
