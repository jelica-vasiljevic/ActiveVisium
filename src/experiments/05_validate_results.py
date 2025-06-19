import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    ConfusionMatrixDisplay
)

from utils import config_utils

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Mapping to normalize prediction labels
DICT_REPLACE = {
    "non neo epithelium": "non neo epithelium",
    "Stroma Fibro IC high": "stroma_fibroblastic_IC high",
    "lamina propria": "lam propria",
    "Tumor &Stroma IC med high": "tumor&stroma IC med to high",
    "Epithelium Lam propria": "epithelium&lam propria",
    "Stroma fibro IC med": "stroma_fibroblastic_IC med",
    "IC Aggregate muscularis or stroma": "IC aggregate_muscularis or stroma",
    "Tumor": "tumor",
    "Connective tissue muscularis": "connective tissue_5_muscularis_IC med to high",
    "Exclude": "exclude",
    "IC Aggregate submucosa": "IC aggregate_submucosa"
}

GT_COLUMN = "Pathologist Annotations"
PRED_COLUMN = "Model Prediction"

# Path to the initial config file (adjust as needed)
INITIAL_CONFIG_PATH = ("../../config_files/experiments/test/multimodal/config_test.json")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """
    Load a configuration JSON from disk using config_utils.
    """
    return config_utils.read_config_from_user_file(path)


def get_run_directories(runs_home: str, test_case: str) -> list:
    """
    List all prediction files and annotation files for a given run directory.
    Returns two sorted lists: predictions, annotation files.
    """
    filenames = os.listdir(runs_home)
    # Prediction files start with "Model_predictions_<test_case>" and exclude "trainingset_predictions"
    preds = [
        fn for fn in filenames
        if fn.startswith(f"Model_predictions_{test_case}")
        and "trainingset_predictions" not in fn
    ]
    preds.sort(reverse=True)

    # Annotation files start with "TrainingAnnotations_"
    annots = [fn for fn in filenames if fn.startswith("TrainingAnnotations_")]
    annots.sort(reverse=True)

    return preds, annots


def read_annotations(path: str) -> pd.DataFrame:
    """
    Read the pathologist annotations CSV.
    """
    return pd.read_csv(path)


def read_predictions(path: str) -> pd.DataFrame:
    """
    Read the model's predictions CSV and normalize labels using DICT_REPLACE.
    """
    df = pd.read_csv(path)
    df[PRED_COLUMN] = df[PRED_COLUMN].replace(DICT_REPLACE)
    return df


def merge_ground_truth_and_preds(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    remove_na: bool = True
) -> pd.DataFrame:
    """
    Merge ground truth and predictions on 'Barcode'. Optionally drop any rows
    where GT is NaN.
    """
    if remove_na:
        before = gt_df.shape[0]
        gt_df = gt_df[~gt_df[GT_COLUMN].isna()]
        removed = before - gt_df.shape[0]
        print(f"Removed {removed} rows with NaN in '{GT_COLUMN}'.")
    merged = pd.merge(gt_df, pred_df, on="Barcode")
    return merged


def compute_and_save_confusion_matrix(
    merged: pd.DataFrame,
    save_path: str
) -> np.ndarray:
    """
    Compute confusion matrix, save as CSV, and return the raw matrix.
    """
    true_labels = merged[GT_COLUMN].tolist()
    pred_labels = merged[PRED_COLUMN].tolist()
    labels = list(set(merged[GT_COLUMN].unique()) | set(merged[PRED_COLUMN].unique()))

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(save_path, "confusion_matrix.csv"), index=True)

    return cm


def display_percentage_in_confusion_matrix(cm: np.ndarray, labels: list):
    """
    Display a normalized confusion matrix as a heatmap (percentages).
    """
    cm_sum = cm.sum(axis=1)
    cm_pct = cm / cm_sum[:, np.newaxis]

    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1
    )
    plt.show()


def compute_and_save_classification_report(
    merged: pd.DataFrame,
    save_path: str
) -> dict:
    """
    Compute classification report, save as CSV, and return the report dict.
    """
    true_labels = merged[GT_COLUMN]
    pred_labels = merged[PRED_COLUMN]
    report = classification_report(true_labels, pred_labels, output_dict=True, digits=3)

    report_df = pd.DataFrame.from_dict(report).transpose()
    report_df.to_csv(os.path.join(save_path, "classification_report.csv"), index=True)

    return report


def plot_time_series(
    values: list,
    xlabel: str,
    ylabel: str,
    xticks: list,
    title: str,
    save_path: str = None
):
    """
    Plot a simple line chart (values vs iterations) and optionally save.
    """
    plt.figure()
    plt.plot(values[::-1], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_fscore_vs_annotations(
    annotations: list,
    scores: list,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str = None
):
    """
    Plot F-score (or weighted F-score) versus number of annotations.
    """
    plt.figure()
    plt.plot(annotations[::-1], scores[::-1], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(annotations)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_misclassified_trend(
    misclassified: list,
    title: str = "Misclassified Trend",
    figsize: tuple = (10, 5)
):
    """
    Plot the number of misclassified samples per iteration with a red dotted linear trend line.
    """
    reversed_vals = misclassified[::-1]
    plt.figure(figsize=figsize)
    plt.plot(reversed_vals, label="Misclassified", marker="o")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Number of Misclassified")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load initial configuration
    print(f"Reading initial config: {INITIAL_CONFIG_PATH}")
    initial_cfg = load_config(INITIAL_CONFIG_PATH)
    test_case = initial_cfg["test_case_name"]
    pathologist_annotations_path = initial_cfg["original_annotations_test_only"]

    # Derive directories
    runs_home_dir = os.path.dirname(initial_cfg["model_savepath"])
    save_graphics_dir = os.path.join(os.path.dirname(runs_home_dir), "graphic_results")
    save_results_dir = os.path.join(os.path.dirname(initial_cfg["model_savepath"]), "results")

    os.makedirs(save_graphics_dir, exist_ok=True)
    os.makedirs(save_results_dir, exist_ok=True)

    print("Current working directory:", os.getcwd())
    print("Runs home directory:", runs_home_dir)
    print("Saving graphics to:", save_graphics_dir)
    print("Saving results to:", save_results_dir)
    print("Test case:", test_case)
    print("Pathologist annotations:", pathologist_annotations_path, "\n")

    # Define which run subdirectories to process (empty string means top-level runs_home_dir)
    run_dirs = [""]  # if you have multiple subfolders, list them here

    # 2. Loop over each run directory
    for run in run_dirs:
        run_dir = os.path.join(runs_home_dir, run)
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            print(f"[WARNING] Config not found for run '{run}': {cfg_path}")
            continue

        
        # Identify prediction and annotation files
        preds, annotations_files = get_run_directories(run_dir, test_case)
        print(f"Run: '{run}' → Predictions found: {len(preds)}, Annotations files: {len(annotations_files)}")
        print(preds, "\n")

        # Counters for plotting
        balanced_acc_scores = []
        macro_f_scores = []
        weighted_f_scores = []
        misclassified_counts = []
        num_annotations = []

        best_report_written = False

        # 3. Iterate over each prediction file
        for idx, pred_fn in enumerate(preds):
            print("=" * 60)
            print(f"Processing: {pred_fn}")

            # Create a subfolder for this iteration’s results
            basename = f"GT_{os.path.splitext(os.path.basename(pathologist_annotations_path))[0]}/" \
                       f"{os.path.splitext(pred_fn)[0]}/"
            iter_save_dir = os.path.join(save_results_dir, basename)
            os.makedirs(iter_save_dir, exist_ok=True)

            # 3.a. Load ground truth and predictions
            gt_df = read_annotations(pathologist_annotations_path)
            pred_df = read_predictions(os.path.join(run_dir, pred_fn))

            # 3.b. Merge and clean
            merged = merge_ground_truth_and_preds(gt_df, pred_df)

            # 3.c. Compute & save confusion matrix
            cm_raw = compute_and_save_confusion_matrix(merged, iter_save_dir)
            print("Confusion matrix saved.")

            # Display normalized confusion matrix (percentage)
            labels_unique = list(merged[GT_COLUMN].unique())
            cm_non_zero = confusion_matrix(
                merged[GT_COLUMN],
                merged[PRED_COLUMN],
                labels=labels_unique
            )
            display_percentage_in_confusion_matrix(cm_non_zero, labels_unique)

            # Also plot sklearn’s ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_non_zero, display_labels=labels_unique)
            disp.plot(xticks_rotation="vertical")
            plt.show()

            # 3.d. Compute balanced accuracy & classification report
            bal_acc = balanced_accuracy_score(merged[GT_COLUMN], merged[PRED_COLUMN])
            balanced_acc_scores.append(bal_acc)
            print(f"Balanced accuracy score: {bal_acc:.4f}")

            with open(os.path.join(iter_save_dir, "balanced_accuracy_score.txt"), "w") as f:
                f.write(str(bal_acc))

            report_dict = compute_and_save_classification_report(merged, iter_save_dir)
            print("Classification report saved.")

            # Track F-scores
            macro_f = report_dict["macro avg"]["f1-score"]
            weighted_f = report_dict["weighted avg"]["f1-score"]
            macro_f_scores.append(macro_f)
            weighted_f_scores.append(weighted_f)

            # Save the “best” report once (first iteration)
            if not best_report_written:
                report_df = pd.DataFrame.from_dict(report_dict).transpose()
                report_df.to_csv(os.path.join(save_graphics_dir, "classification_report_separate.csv"), index=True)
                best_report_written = True

            # 3.e. Count misclassified
            miscls = merged[merged[GT_COLUMN] != merged[PRED_COLUMN]].shape[0]
            misclassified_counts.append(miscls)

            print(f"Misclassified count: {miscls}")
            print("=" * 60, "\n")

        # 4. Count annotations per iteration
        for annot_fn in annotations_files:
            df = pd.read_csv(os.path.join(run_dir, annot_fn))
            df = df[df[GT_COLUMN] != "help"]  # filter out “help” rows
            num_annotations.append(df.shape[0])

        # 5. Plot time-series of metrics
        iterations = list(range(len(balanced_acc_scores)))
        plot_time_series(
            balanced_acc_scores,
            xlabel="Iteration",
            ylabel="Balanced Accuracy",
            xticks=iterations,
            title="Balanced Accuracy per Iteration",
            save_path=os.path.join(save_graphics_dir, f"{os.path.basename(runs_home_dir)}_balanced_accuracy.png")
        )

        plot_time_series(
            macro_f_scores,
            xlabel="Iteration",
            ylabel="Macro F1 Score",
            xticks=iterations,
            title="Macro F1 Score per Iteration",
            save_path=os.path.join(save_graphics_dir, f"{os.path.basename(runs_home_dir)}_macro_f1_scores.png")
        )

        # 6. Plot F-score vs. Number of Annotations
        if len(num_annotations) == len(macro_f_scores):
            plot_fscore_vs_annotations(
                annotations=num_annotations,
                scores=macro_f_scores,
                xlabel="Number of Annotations",
                ylabel="Macro F1 Score",
                title="Macro F1 Score vs Number of Annotations",
                save_path=os.path.join(save_graphics_dir, f"{os.path.basename(runs_home_dir)}_f1_vs_annotations.png")
            )
        else:
            print("[WARNING] Annotations count and F-scores length mismatch.")

        # 7. Plot weighted F-score vs. Number of Annotations
        if len(num_annotations) == len(weighted_f_scores):
            plot_fscore_vs_annotations(
                annotations=num_annotations,
                scores=weighted_f_scores,
                xlabel="Number of Annotations",
                ylabel="Weighted F1 Score",
                title="Weighted F1 Score vs Number of Annotations",
                save_path=os.path.join(save_graphics_dir, f"{os.path.basename(runs_home_dir)}_weighted_f1_vs_annotations.png")
            )
        else:
            print("[WARNING] Annotations count and weighted F-scores length mismatch.")

        # 8. Summarize per-iteration info
        for i, annot_fn in enumerate(annotations_files):
            print(
                f"Iteration {i}: "
                f"Annotation file: {annot_fn}, "
                f"#Annotations: {num_annotations[i] if i < len(num_annotations) else 'N/A'}, "
                f"Balanced Acc: {balanced_acc_scores[i] if i < len(balanced_acc_scores) else 'N/A'}, "
                f"Macro F1: {macro_f_scores[i] if i < len(macro_f_scores) else 'N/A'}, "
                f"Misclassified: {misclassified_counts[i] if i < len(misclassified_counts) else 'N/A'}"
            )

        # 9. Print total ground-truth annotations
        gt_df_final = read_annotations(pathologist_annotations_path)
        print(f"Total GT annotations: {gt_df_final.shape[0]}")

        # 10. Plot misclassified trend with linear offsets
        plot_misclassified_trend(
            misclassified_counts,
            title="Misclassified Samples Over Iterations"
        )

        print(weighted_f_scores)


if __name__ == "__main__":
    main()
