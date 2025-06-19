# 🧜 ActiveVisium: Leveraging Active Learning to Enhance Manual Pathologist Annotation in 10x Visium Spatial Transcriptomics Experiments

This is the official implementation of *ActiveVisium: Leveraging Active Learning to Enhance Manual Pathologist Annotation in 10x Visium Spatial Transcriptomics Experiments*, as presented at ECML-PKDD 2025 (Applied Data Science Track).

**ActiveVisium** is a framework designed to enhance manual pathologist annotation in 10x Visium experiments. It implements a human-in-the-loop active learning approach to efficiently guide expert annotation at the spot level, significantly reducing annotation time.

ActiveVisium leverages tissue morphology (from histological images) and, optionally, gene expression data to identify the most informative spots for manual labeling. The framework iteratively trains a classifier on the current set of annotated spots, predicts labels for the remaining spots, and selects new spots for expert annotations using uncertainty and diversity-based sampling strategies. This process continues until the annotation is sufficiently complete or meets user-defined criteria.

---


## Table of Contents

- [Project Highlights](#project-highlights)
- [Data Availability](#data-availability)
- [Installation](#installation)
- [Quick Start: Test Whole Pipeline](#quick-start-test-whole-pipeline)
- [Quick Start: Practical Use Case](#quick-start-practical-use-case)
- [Running ActiveVisium on New Samples](#running-activevisium-on-new-samples)
  - [Preparing Data](#preparing-data)
  - [Unimodal Setting (Morphology Only)](#unimodal-setting-morphology-only)
  - [Multimodal Setting (Morphology and Gene Expression)](#multimodal-setting-morphology-and-gene-expression)
  - [Initial Step for Active Learning](#initial-step-for-active-learning)
- [Reproducing Results from the Paper](#reproducing-results-from-the-paper)
- [Testing on Pre-Annotated Datasets (Optional)](#testing-on-pre-annotated-datasets-optional)
- [Citation](#citation)
- [Contact](#contact)




---

## Project Highlights

- 🧠 **Active Learning:** Iterative selection of the most informative spots for annotation.
- 🧬 **Multimodal Support:** Integrates gene expression and histology features.
- 🧩 **Flexible Architecture:** Compatible with state-of-the-art (SOTA) foundational models; easily extensible to emerging models.
- 📈 **Annotation Efficacy:** Significantly reduces pathologist workload by focusing annotation efforts where they are most impactful.

---

## Data Availability

All data used in the paper are available for download from [Zenodo](https://zenodo.org/records/15625540), including OpenSlide-compatible whole-slide images, patch-level embeddings generated using the multiple foundational models, intermediate results, expert pathologist annotations, and trained model checkpoints.

⚠️ **Caution:** These folders can be large in size and may require substantial storage and memory resources to handle effectively.

---
## Installation

### Set Up Anaconda Environment

Ensure you have Anaconda installed, then create and activate an environment for ActiveVisium:

```
conda create --name activevisium_env python=3.8
conda activate activevisium_env
```

### Configure Weights & Biases (WandB) and Hugging Face

ActiveVisium uses Weights & Biases (WandB) for tracking training and validation. To enable these features, set the following environment variables:

```
export WANDB_API_KEY="YOUR_API_KEY"
export WANDB_BASE_URL="BASE_URL"
```

To have access to foundational models, obtain and set a Hugging Face token:

```
export HUGGINGFACE_TOKEN="YOUR_TOKEN"
```
*You should also place this token in file src/global_constants.py*


Output: Each experiment run will be logged in the WandB directory. A separate run will be created for each active learning iteration.

---


## Quick Start: Test whole pipeline

1. Download example data (breast cancer sample) from Zenodo and place it in the parent directory of the source code, under the directory `data`.
2. Adjust the script in `scripts/run_activevisium_test_case.sh` to include the corresponding API keys.
3. Run the script:

    ```
    cd scripts; 
    ./run_activevisium_test_case.sh
    ```
    The results will be stored inside the `test_data` folder.

4. (Optional) Visualize results using `05_validate_results.py` to obtain confusion matrices and various scores over active learning iterations.

## Quick Start: Practical Use Case

**Requirements:**  
- Loupe Browser (for annotation and visualization)

1. **Open & Group**  
   - Launch Loupe Browser and open `.cloupe` file from test_data folder.  
   - Create a new group named `Pathologist Annotations`.  

2. **Annotate Spots**  
   - Add broad labels to some of the spots
   - For more details, see “Identifying Initial Set of Annotations Automatically” (example: `../doc/example_image.png`).  

3. **Export Annotations**  
   - In Loupe, export your annotations (excluding any unlabeled spots) as `Pathologist_Annotations.csv`.  
   - Place this file in the `ActiveVisium_practical_usecase/` folder, which should be in the same directory as project `ActiveVisium_paper/`

4. **Run the Pipeline**  
   - Edit `scripts/ActiveVisium_practical_usecase.sh` to point to the correct data and output directories.  
   - Execute the script:  
     ```bash
     cd scripts;
     ./ActiveVisium_practical_usecase.sh
     ```  

5. **Review Outputs**  
   After the script finishes, check the `ActiveVisium_practical_usecase/` folder for:  
   - `ActiveVisium_annotations.csv` — all annotated spots  
   - `Predictions_with_help.csv` — spots suggested for the next iteration (labelled “help”)  
   You can load these CSVs into Loupe Browser (see example: `doc/example_predictions.png`).  

_For the next active learning iterations, refer to section 4, “Run Active Learning and Update Annotations.”_
---


## Running ActiveVisium on New Samples

### Preparing Data

Make sure that in the directory that contains data on which you would like to try ActiveVisium you have the following files:
- Whole-Slide-Image (WSI) used for generating SpaceRanger output. In case lower magnification is used during SpaceRanger, make sure to provide proper scaling factor when extracting patches.
- `spatial` folder and corresponding `tissue_positions_list.csv` file.

Prepare the config file. Make sure that `dataset` and `test_case_name` are *THE SAME*. In case of transfer learning, you should modify `test_case_name` to the name of the dataset to which you would like to apply it. But for training purposes, they should be the same.

#### 1. Extract patches that correspond to the spots:

```
cd data;
python PatchExtractor.py -img ../../../data/breast_test_from_scratch/CytAssist_FFPE_Human_Breast_Cancer_tissue_image_openslide.tif ;
cd ..
```

If the image used during SpaceRanger processing is a downsampled version of the WSI you’re using for patch extraction, make sure to specify the correct image scaling factor. For example, if SpaceRanger was run on a 10x image and you are extracting patches from a 20x WSI, set:

```
-scalingFactor 0.5
```

⚠️ The script will also generate a `debug_spots.png` file, showing the low-resolution image overlaid with spots. Use this as a quick visual check to ensure that spot coordinates are mapped correctly (e.g., no row/column inversion).

⚠️ Make sure to adjust parameters in config files to use the extracted patches/representations - `patch_dir`, `test_patch_dir` and `test_patch_size`.

#### 2. Generate embeddings from a foundational model (e.g., UNI):

⚠️ Make sure to have `HUGGINGFACE_TOKEN` properly set up and granted access to the model.

```
cd models;
python extract_features.py ../../config_files/experiments/breast_test_from_scratch/config_example_1.json ;
cd ..
```

The extracted features will be stored in the same directory where patches are extracted as an `.h5` file.

During feature extraction, data augmentation is applied. As a result, the process is stochastic and can yield slightly different representations for the same input. To address this, users can specify the number of repetitions as the second argument, which controls how many times features are extracted (with augmentation). By default, this value is set to 10. The filename indicates how many repetitions are applied, e.g., `UNI_10_representations.h5`. The script will create `n_repetitions` representations for the whole dataset (using augmentation, train set) and one representation for the whole dataset without augmentation (valid set). This approach ensures that we have feature representations for all data points, regardless of whether they will be used for training (and thus augmented) or validation (not augmented), since the specific usage of each data point may not be known in advance. For computational reasons, feature extraction is performed only once, and therefore representations for all spots are generated in this way.

In practice, we have found that foundational models—due to their training on heavily augmented datasets—produce stable features across repetitions, with minimal variation. However, if you observe significant variability in your dataset, consider increasing the number of repetitions for greater stability. Additionally, we provide a visualization script to help you inspect the extracted feature representations.

---

### Multimodal Setting (Morphology and Gene Expression)

The script `notebooks/01_Prepare_Genomic_Data.ipynb` computes the top 1,000 highly variable genes from the input dataset and saves the resulting AnnData object to the data directory for downstream multimodal analysis. If you plan to run ActiveVisium in a multimodal setting across multiple samples (e.g., for annotation transfer tasks), it is recommended to harmonize the data using Harmony. An example of how to perform this integration is provided in the script `notebooks/02_Harmony_integration.ipynb`.

---

## Active Learning

To initiate active learning training, the expert must provide an initial set of annotated samples, ensuring that each class present in the dataset is represented by at least one labeled spot. While it is theoretically possible to introduce new classes later in the training process, our practical experience suggests that defining all relevant classes in advance leads to higher annotation quality and greater consistency. We recommend that users take time to examine the tissue and anticipate the types of annotations that may be relevant for downstream analysis.

### 1. Identifying Initial Set of Annotations Automatically

ActiveVisium can assist in selecting the initial set of spots for annotation by automatically identifying the most diverse candidates for a pathologist to label first. This selection can be performed by applying k-means clustering to the feature representation space of the tissue spots, and then choosing the top K most distinct spots, where K is the desired number of initial annotations.

This approach typically results in a diverse subset that captures several meaningful tissue classes—often aligning with what a pathologist might select manually. However, it may not cover all classes upfront, as subtle differences identifiable by experts may not yet be separable in the feature space.

Alternatively, the initial annotation set can be selected randomly. The selection strategy is configurable by setting the `initial_spot_selection` parameter in the config file to either:

- `foundational_model_diversity_$MODEL` (e.g., `foundational_model_diversity_UNI`) for clustering-based selection, or
- `random` for random sampling.

To obtain the initial annotated dataset, run the following script:

```
python 01_Inital_clustering_and_selection_practical_usage.py ../../config_files/experiments/breast_test_from_scratch/config_example_1.json
```

A file containing the selected spots will be generated in the dataset directory under the name: `TrainingAnnotations$diversity.csv`, for example, `TrainingAnnotationsfoundational_model_diversity_UNI.csv`.  
This file includes the selected barcodes with a placeholder annotation marked as `"help"` to indicate spots that should be annotated by a pathologist. An example of the file content:

```
Barcode,Pathologist Annotations
CTATAGGTTGATGCCT-1,help
TCCTTACTCGCAATGA-1,help
AACTTCACAATAACTG-1,help
TCCAACGTGGTTGTTC-1,help
```

---

### 2. Open Loupe Browser and Import Initial Annotations

Open **Loupe Browser** and load the `.loupe` file generated by SpaceRanger for your dataset.  
Then, import the `TrainingAnnotations*.csv` file (e.g., `TrainingAnnotationsfoundational_model_diversity_UNI.csv`) as the initial spot annotation file.

---

### 3. Annotate Spots in Loupe Browser

Spot annotation is a **critical** and often **time-consuming** step in practice. Please follow these guidelines carefully:

1. Use the **rectangle** or **freehand** selection tools to annotate individual spots and define classes.

2. If multiple spots belong to the same class:
   - First, annotate a **single spot** to create the class label.
   - Then, use the **brush tool** to annotate additional spots belonging to the same class.
   - ⚠️ **Important:** Before using the brush tool, ensure the correct class is selected in the left-hand side panel.
   - ⚠️ **Caution:** Use the brush tool **very carefully**—you can unintentionally annotate spots simply by hovering over them.

3. Once all relevant classes are annotated:
   - **Delete the placeholder `"help"` class** from the left-hand side class list.
   - Go to export button and when prompted, **choose the option `"Exclude unlabeled"`**.
   - Save annotations in a directory of your choice under the name `Pathologist_Annotations.csv`.

---

### 4. Run Active Learning and Update Annotations

#### 4.1 Start the Active Learning Script

Run the following script from within the `scripts/` directory:

```
./ActiveVisium_practical_usecase.sh
```

This will:
- Train the model using the current annotation set.
- Generate a prediction file named: `Predictions_with_help__<TIMESTAMP>.csv` in the same folder where the initial annotations were saved.
- Generate a file containing all spots annotations named `ActiveVisium_annotations.csv`.

---

#### 4.2 Import Predictions into Loupe Browser

Open Loupe Browser and import the `Predictions_with_help__<TIMESTAMP>.csv` file.

This file includes:
- Model-predicted class labels
- A special `"help"` class, highlighting spots where the model is uncertain and requesting expert input

---

#### 4.3 Annotate "Help" Spots

⚠️ **Important:** In the **first 2-3 rounds** of active learning:
- Only annotate spots labeled as `"help"`  
- This ensures the model focuses on resolving areas of uncertainty during its early learning phase
- While not strictly required, deviating from this recommendation will make the annotation process resemble random sampling, which—as demonstrated in our paper—results in inferior model performance compared to following the suggested approach.

After these initial rounds:
- The pathologist may also choose to review and correct other model predictions if needed (e.g., model mistakes).

---

#### 4.4 Export Updated Annotations

After annotating the `"help"` spots, export the updated annotation file as before.  
Repeat the active learning cycle as needed.

---

###  Transfer Learning

To generate predictions on a dataset different from the one used during training:

1. **Update the Configuration File**  
   Set the `test_case_name` field to the name of the new test dataset.

2. **Ensure Dataset Location Consistency**  
   Both the training and new test datasets must reside in the same `base_path`,  
   as the code expects to find them there.

3. **Output Structure**  
   The model will write predictions for the test dataset to the same output directory  
   used during training on the training dataset. The prediction file will be named: ```Model_predictions_${test_case_name}.csv```. 



## Reproducing Paper Results

The `notebooks/paper_results/` folder includes Jupyter notebooks that reproduce all quantitative results from the paper.

### Data Setup

1. Download the dataset from Zenodo:  
   https://zenodo.org/records/15625540  
2. Extract the downloaded archive and place the resulting `data/` directory **next to** this repository’s source code (i.e., in its parent directory).




---

## Testing on Pre-Annotated Datasets (Optional)

To benchmark ActiveVisium on datasets with existing annotations:

1. Prepare your data as for new samples.
2. Use scripts in `src/dataset_split_first_time_exp/` to:
    - Split into train/val/test sets.
    - Define the initial annotation set.
    - Map ground-truth annotations.
3. Additional scripts are in the experiments folder


This ensures a consistent and fair baseline for evaluating active learning improvements, including when leveraging novel foundational models for feature extraction or alternative model architectures.

---

## Citation

If you use ActiveVisium in your research, please cite:

```
@inproceedings{activevisium2025,
  title={ActiveVisium: Leveraging Active Learning to Enhance Manual Pathologist Annotation in 10x Visium Spatial Transcriptomics Experiments},
  author={J. Vasiljevic, I. B. Veiga, K. Hahn, P. Schwalie, A. Valdeolivas},
  booktitle={ECML-PKDD 2025, Applied Data Science Track},
  year={2025}
}
```

---

## Contact

For questions or contributions, please open an issue.
