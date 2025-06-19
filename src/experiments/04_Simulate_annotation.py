# add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import pandas as pd
from utils import config_utils
import datetime
import shutil
from global_constants import gt_column_name

config_file_path =  sys.argv[1] 
my_configs = config_utils.read_config_from_user_file(config_file_path)

debug = False
remove_barcodes_not_in_gt = False


to_annotate = pd.read_csv(my_configs['loupe_csv_file'], index_col=0)


#print number of duplicates in to_annotate index
print(f'Number of duplicates: {to_annotate.index.duplicated().sum()}')

# read original annotations
pathological_annotations = pd.read_csv(my_configs['original_annotations_test_only'], index_col=0)

# replace "help" values in to_annotate with values from pathological_annotations based on barcode. 
merge = to_annotate.merge(pathological_annotations, on='Barcode', how='left',suffixes=('', '_annotated'))
merge[gt_column_name] = merge.apply(lambda row: row[gt_column_name+'_annotated'] if row[gt_column_name] == 'help' else row[gt_column_name], axis=1)

# remove Pathologist Annotations_annotated column
merge = merge.drop(columns=[gt_column_name+'_annotated'])


# check if there is NaN values in Pathologist Annotations column
print(f'Number of NaN values in Pathologist Annotations: {merge[gt_column_name].isna().sum()}')

# replace NaN values with "excluded". Take into account that sometimes exclude can be written in different ways
exclusion_labels = pathological_annotations[gt_column_name].unique()
exclusion_label = [label for label in exclusion_labels if 'exclude' in label.lower()]
# check if there is any exclusion label
if len(exclusion_label) > 0:
    exclusion_label = exclusion_label[0]
else:
    exclusion_label = 'exclude'

# replace NaN values with exclusion_label
print(f'Replacing NaN values with {exclusion_label}')
merge[gt_column_name] = merge[gt_column_name].fillna(exclusion_label)

# print how many duplicates indexes are there
print(f'Number of duplicates: {merge.index.duplicated().sum()}')


# remove duplicates based on index
merge = merge[~merge.index.duplicated(keep='first')]

print("\033[91mNumber of barcodes that don't exist in pathological_annotations: {}\033[00m".format((~merge.index.isin(pathological_annotations.index)).sum()))
print(merge[~merge.index.isin(pathological_annotations.index)].index)

if remove_barcodes_not_in_gt:
    # print message in red color that these spots are removed from merge
    print("\033[91m Removing these spots....\033[00m")
    # remove these barcodes from merge
    merge = merge[merge.index.isin(pathological_annotations.index)]
else:
    print("\033[91m These spots are replaced with 'excluded'\033[00m")

# make backup of my_configs['loupe_csv_file'] by adding timestamp to the end of the file name
if not debug:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # get the name of the file without extension
    filename = os.path.splitext(my_configs['loupe_csv_file'])[0]

    shutil.copy(my_configs['loupe_csv_file'], filename +"_"+ timestamp+".csv")

    #save to_annotate to csv
    merge.to_csv(my_configs['loupe_csv_file'])
    print(f'Annotations saved to {my_configs["loupe_csv_file"]}')
    os._exit(0)