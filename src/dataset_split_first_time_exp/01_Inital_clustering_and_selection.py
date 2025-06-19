
"""
This script performs initial clustering and spot selection based on specified configuration file. 

Usage:
python 01_Inital_clustering_and_selection.py <config_file_path>

Args:
- config_file_path: Path to the configuration file containing the necessary parameters for the script.

Output:
- initial_clusters.csv: CSV file containing the cluster labels for each barcode.
- TrainingAnnotations$method.csv: CSV file containing the selected barcodes.

"""
import numpy as np
import pandas as pd
import sys
# add parent directory to path
import os
sys.path.append(os.path.dirname(os.getcwd()))
from utils import config_utils
import  re
from albumentations import *
import sys
from active_learning_methods.diversity import *

def get_random_barcodes(image_barcodes, total_number_of_samples):
    """
    Randomly selects a specified number of barcodes from a given list of image barcodes.

    Parameters:
    image_barcodes (list): A list of image barcodes.
    total_number_of_samples (int): The total number of barcodes to be selected.

    Returns:
    list: A list of randomly selected barcodes.
    """
    print("Random selection")
    images_selected = np.random.choice(image_barcodes, total_number_of_samples, replace=False)
    barcodes_selected = [re.findall(r'[TGCA]+\-[0-9]',image_name)[0] for image_name in images_selected]
    return barcodes_selected



def main(config_file_path):
    
    my_configs = config_utils.read_config_from_user_file(config_file_path)
    run_save_dir = os.path.join(my_configs['base_path'], my_configs['dataset'])
    # create directory run_save_dir if it does not exist
    os.makedirs(run_save_dir, exist_ok=True)
    print("\033[91mrun_save_dir: {}\033[00m".format(run_save_dir))


    # ************************************************
    # Prepare data
    # ************************************************
    
    patch_dir = my_configs['patch_dir']
    
    # Get list of image files in the patch directory
    image_files = os.listdir(patch_dir)
    # Extract barcodes from image filenames
    image_barcodes = [(filename.split('_')[1]).split('.')[0] for filename in image_files if filename.startswith('patch_')]


    # check if training data file exists
    if os.path.exists(my_configs['train_data']):        
        # Expperimental results
        # ************************************************
        # Load barcodes from training pool and filter out images that are not in the pool
        # ************************************************

        training_pool = pd.read_csv(my_configs['train_data'], index_col=0)
        image_barcodes = [barcode for barcode in image_barcodes if barcode in training_pool.index]
    
    barcodes_selected = None
    clusters = None

    if my_configs['initial_spot_selection'] == 'random':

        barcodes_selected = get_random_barcodes(image_barcodes, my_configs['total_number_of_samples_for_all_clusters'])
    
    elif 'foundational_model_diversity' in my_configs['initial_spot_selection']:
        
        # print in yellow
        print("\033[93mFoundational model diversity sampling nased on model",  my_configs['hdf5_file'],"\033[00m")
        
        
        images_dataframe = pd.DataFrame(image_barcodes, columns=['Barcode'])
        # set Barcodes as index
        images_dataframe.set_index('Barcode', inplace=True)
        
        barcodes_selected, clusters = diversity_sampling_based_on_pretrained_model_representations(images_dataframe, my_configs['hdf5_file'],my_configs['total_number_of_samples_for_all_clusters'],device='cuda')
        # save clusters to csv
        clusters.to_csv(os.path.join(run_save_dir, f"initial_clusters_{my_configs['initial_spot_selection']}.csv"), index=True)

    else:
       raise ValueError(f"Initial spot selection {my_configs['initial_spot_selection']} method not recognized")
    
    if clusters is not None:
        # save clusters to csv
        clusters.to_csv(os.path.join(run_save_dir, f"initial_clusters_{my_configs['initial_spot_selection']}.csv"), index=True)
        
    assert barcodes_selected is not None, "Barcodes selected is None"
    # make dataframa from selected barcodes
    todo_array = ["help" for _ in barcodes_selected]
    data = {'Barcode': barcodes_selected, 'Pathologist Annotations': todo_array}
    df = pd.DataFrame(data)
    df.set_index('Barcode', inplace=True)   
    df.to_csv(os.path.join(run_save_dir,"TrainingAnnotations"+my_configs['initial_spot_selection']+".csv"), index=True)

    print("Selected barcodes saved to CSV file.")
    print(os.path.join(run_save_dir,"TrainingAnnotations"+my_configs['initial_spot_selection']+".csv"))


if __name__ == "__main__":
    config_file_path = sys.argv[1]
    main(config_file_path)
    print("Done!")
    os._exit(0)  # Exit the script


