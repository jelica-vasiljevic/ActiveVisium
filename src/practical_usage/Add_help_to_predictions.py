import pandas as pd
import sys
# add parent directory to sys.path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config_utils
import os
import shutil

# print in purple that the script is starting
print("\033[95m ******* Adding help into predictions ******* \033[0m")

config_file_path = sys.argv[1]
filename_to_save = sys.argv[2]

my_configs = config_utils.read_config_from_user_file(config_file_path)
run_dir = os.path.join(os.path.dirname(my_configs['model_savepath']))

list_of_files = [f for f in os.listdir(run_dir) if f.startswith(f'Model_predictions_{my_configs["test_case_name"]}_')]

# if list_of_files is empty, print in red that there are no files and exit
if not list_of_files:
    print("\033[91m No prediction files found \033[0m")
    sys.exit(0)
    
list_of_files_sorted = sorted(list_of_files, reverse=True)
latest_file_path = os.path.join(run_dir, list_of_files_sorted[0])

# read the latest prediction file
predictions_df = pd.read_csv(latest_file_path, index_col=0)

# read TrainingAnnotations.csv
annotations_df = pd.read_csv(my_configs['loupe_csv_file'], index_col=0)

# replace predictions with help
predictions_df.loc[annotations_df.index, 'Model Prediction'] = annotations_df['Pathologist Annotations']

# save the new predictions as a new file Predictions_with_help.csv and add the timestamp
timestamp = list_of_files_sorted[0].split('_')[-1].split('.')[0]
new_file_path = os.path.join(run_dir, f'{filename_to_save}_{timestamp}.csv')
predictions_df.to_csv(new_file_path, index=True)

# store latest predictions as ActiveVisium_annotations.csv
active_visium_path = os.path.join(run_dir, 'ActiveVisium_annotations.csv')
predictions_df = pd.read_csv(latest_file_path, index_col=0)
# change the name of the column to 'Active Visium Annotations'
predictions_df.rename(columns={'Model Prediction': 'Active Visium Annotations'}, inplace=True)
predictions_df.to_csv(active_visium_path, index=True)

