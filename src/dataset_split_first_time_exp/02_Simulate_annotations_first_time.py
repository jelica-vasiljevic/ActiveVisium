"""
This script simulates annotations by replacing 'help' entries in a spot-level CSV 
with ground truth annotations from the pathologist annotation file.

It ensures that at least one example of each annotation class present in the ground 
truth is included in the output (the initial training annotation set), even if 
they were not selected initially.

Intended to be run only once, after executing 
`01_Initial_annotation_and_selection.py` during the initial setup of ActiveVisium 
for experimental workflows.

Usage:
    python 02_Simulate_annotations_first_time.py <config_file_path>
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from utils import config_utils
import datetime
import shutil
from global_constants import *

def main(config_file_path):

    my_configs = config_utils.read_config_from_user_file(config_file_path)

    # check if the file exists
    run_save_dir = os.path.dirname(my_configs['original_annotations_test_only'])
    file_path = os.path.join(run_save_dir,"TrainingAnnotations"+my_configs['initial_spot_selection']+".csv")
    to_annotate = pd.read_csv(file_path, index_col=0)
    save_file=file_path

    #print number of duplicates in to_annotate index
    num_duplicates = to_annotate.index.duplicated().sum()
    print(f'Number of duplicates in to_annotate index: {num_duplicates}')
    assert num_duplicates == 0, "There are duplicated barcodes for annotation! Please remove them before running this script."

    # read original annotations
    pathological_annotations = pd.read_csv(my_configs['original_annotations_test_only'], index_col=0)
    print(pathological_annotations)

    # replace "help" values in to_annotate with values from pathological_annotations based on barcode. 
    merge = to_annotate.merge(pathological_annotations, on='Barcode', how='left',suffixes=('', '_annotated'))
    merge[gt_column_name] = merge.apply(lambda row: row[gt_column_name+'_annotated'] if row[gt_column_name] == 'help' else row[gt_column_name], axis=1)

    # remove Pathologist Annotations_annotated column
    merge = merge.drop(columns=[gt_column_name+'_annotated'])

    # unique classes in Pathologist Annotations
    unique_classes = merge[gt_column_name].unique()
    unique_classes_annotations = pathological_annotations[gt_column_name].unique()

    for class_name in unique_classes_annotations:
        if class_name not in unique_classes:
            print(f'Class {class_name} is not present in to_annotate')
            # select random barcode from pathological_annotations with class_name
            random_barcode = pathological_annotations[pathological_annotations[gt_column_name] == class_name].sample().index[0]
            # add this point to merge (barcode, class_name)
            merge.loc[random_barcode] = class_name
            
    # check now if all classes are present in merge
    unique_classes = merge[gt_column_name].unique()

    if set(unique_classes) == set(unique_classes_annotations):
        print('All classes are present in merge')

    # print how many duplicates indexes are there
    print(f'Number of duplicates: {merge.index.duplicated().sum()}')

    # remove duplicates based on index
    merge = merge[~merge.index.duplicated(keep='first')]

    print("\033[91mNumber of barcodes that don't exist in pathological_annotations: {}\033[00m".format((~merge.index.isin(pathological_annotations.index)).sum()))
    print(merge[~merge.index.isin(pathological_annotations.index)].index)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # get the name of the file without extension
    filename = os.path.splitext(save_file)[0]
    shutil.copy(save_file, filename +"_"+ timestamp+".csv")

    #save to_annotate to csv
    merge.to_csv(save_file)
    print(f'Annotations saved to {save_file}')


if __name__ == "__main__":
    config_file_path = sys.argv[1]
    main(config_file_path)
    print("Done!")
    os._exit(0)  # Exit the script



os._exit(0)