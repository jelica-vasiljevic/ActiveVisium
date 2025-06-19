import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config_utils
import shutil


if __name__ == "__main__":
    print("\033[95m ******* Forming a training set ******* \033[0m")

    config_file_path = sys.argv[1]
    current_annotation_file = sys.argv[2]
    file_pattern = sys.argv[3]
    
    my_configs = config_utils.read_config_from_user_file(config_file_path)
    run_dir = os.path.join(os.path.dirname(my_configs['model_savepath']))

    # create run directory if it does not exist
    if not os.path.exists(run_dir):
        print(f"Creating directory {run_dir}")
        os.makedirs(run_dir)
    

    
    # read the current annotation file
    print(f"Reading annotations from {current_annotation_file}")
    current_annotation_df = pd.read_csv(current_annotation_file)
    current_annotation_df.set_index('Barcode', inplace=True)


    # check if some of values in column 1 is "help"
    if any(current_annotation_df.iloc[:, 0].str.contains('help', na=False)):
        print("\033[91m Found 'help' in the annotations \033[0m")
        print("You did not provide any new annotations")
        sys.exit(0)   

    # copy current_annotation_file to run_dir, taking current timestamp
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_annotation_file = os.path.join(run_dir, f'File_provided_at_{timestamp}.csv')
    shutil.copy(sys.argv[2], current_annotation_file)

    list_of_provider_files = [f for f in os.listdir(run_dir) if f.startswith('File_provided_at_')]
    if len(list_of_provider_files) > 1:
        # compare last two files
        list_of_provider_files_sorted = sorted(list_of_provider_files, reverse=True)
        previous_file = os.path.join(run_dir, list_of_provider_files_sorted[1])
        current_file = os.path.join(run_dir, list_of_provider_files_sorted[0])

        previous_df = pd.read_csv(previous_file)
        current_df = pd.read_csv(current_file)

        # check if the two files are the same
        if previous_df.equals(current_df):
            print("\033[91m The two files are the same \033[0m")
            print("You did not provide any new annotations")
            sys.exit(0)


    list_of_files = [f for f in os.listdir(run_dir) if f.startswith(file_pattern)]

    # if list_of_files is empty, print in red that there are no files and exit
    if not list_of_files:
        print("\033[91m No prediction files found \033[0m")
        # save the current annotation file
        print(f"Copying {current_annotation_file} to {my_configs['loupe_csv_file']}")
        shutil.copy(current_annotation_file, my_configs['loupe_csv_file'])
        sys.exit(1)

    list_of_files_sorted = sorted(list_of_files, reverse=True)
    latest_file_path = os.path.join(run_dir, list_of_files_sorted[0])



    # read the latest prediction file
    print(f"Reading predictions from {latest_file_path}")
    predictions_df = pd.read_csv(latest_file_path)
    # set the Barcode column as the index
    predictions_df.set_index('Barcode', inplace=True)
 

    # Find differences between the two files by merging them
    diff = predictions_df.merge(current_annotation_df, on='Barcode', suffixes=(' predictions', ' annotation'))

    # find the rows where the model prediction is different from the current annotation
    diff = diff[diff['Model Prediction predictions'] != diff['Model Prediction annotation']]

    # cehck if diff is empty
    if diff.empty:
        print("\033[91m No differences found \033[0m")
        print("You did not provide any new annotations")
        sys.exit(0)
    else:
        # print how many differences were found
        print(f"Found {len(diff)} differences")

    new_annotations = diff['Model Prediction annotation']

    # find previous training set
    prev_annotations = pd.read_csv(my_configs['loupe_csv_file'], index_col=0)

    # merge the new annotations with the previous training set
    new_training_set = prev_annotations.merge(new_annotations, left_index=True, right_index=True, how='outer')

    # accept Model Prediction annotation where it not NaN
    new_training_set['Pathologist Annotations'] = new_training_set['Model Prediction annotation'].fillna(new_training_set['Pathologist Annotations'])
    # remove raws where Pathologist Annotations is 'help'
    new_training_set = new_training_set[new_training_set['Pathologist Annotations'] != 'help']
    # remove Model Prediction annotation column
    new_training_set.drop(columns='Model Prediction annotation', inplace=True)
    # save the new training set

    # make a copy of the previous training set by adding the current time at the end of the file name
    # my_configs['loupe_csv_file']
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    shutil.copy(my_configs['loupe_csv_file'], my_configs['loupe_csv_file'] + '_' + timestamp + '.csv')
    # save the new training set to the previous training set file
    new_training_set.to_csv(my_configs['loupe_csv_file'])

    print("\033[95m ******* DONE ******* \033[0m")

    sys.exit(1)