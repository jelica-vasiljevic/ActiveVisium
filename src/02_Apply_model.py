# Import standard libraries
import os
import pandas as pd
import re
import json
import time
import sys
from datetime import datetime
import torch.nn.functional as F
import torch
import math

# Import third-party libraries
from tqdm import tqdm

# Import local modules
from utils import config_utils
from models import model_utils
from data.Dataset import PrecomputedImageDataset, PrecomputedImageDataset_Multimodal
from data import Preprocessing
from active_learning_methods.uncertainty import calculate_uncertainty
from active_learning_methods.diversity import diversity_sampling
import logging
import argparse
from global_constants import gt_column_name,colname_in_pred
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

class ApplicationManager:
    def __init__(self, config_file_path, debug=False):
        """
        Initialize the ApplicationManager.

        Loads configuration files, sets up device and paths, prepares the dataset,
        and initializes the model for prediction or active learning.

        Parameters
        ----------
        config_file_path : str
            Path to the user configuration file.
        debug : bool, optional
            If True, enables debug mode. Default is False.

        Raises
        ------
        SystemExit
            If the maximum number of active learning runs is exceeded.
        """
        self.config_file_path = config_file_path
        self.debug = debug
        self.configs_test = self._load_config()
        self.configs_model = self._load_model_config()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.run_save_dir = os.path.join(os.path.dirname(self.configs_model['model_savepath']))
        self.mapping = self._load_mapping()
        
        if self.configs_model['num_active_learning_runs'] >= 11:
            print("Reached maximum number of active learning runs")
            os._exit(0)

        # determine mode
        self.mode = 'active_learning'
        if os.path.exists(self.configs_model['loupe_csv_file']) and os.path.exists(self.configs_model['train_data']) and os.path.samefile(self.configs_model['loupe_csv_file'], self.configs_model['train_data']):
            print("This is fully supervised setting, so no new samples are added to the training set.")           
            self.mode = 'fully_supervised'
     
               
        # Set model parameters
        self.AL_model = None
        self.number_of_classes = self.configs_model['train_num_classes']
        self.model_path = self.configs_model['model_savepath']
        self.autoencoder_model_path = self.configs_model.get('autoencoder_h5', None)
        self.select_uncertainty = self.configs_model['uncertainty_metric']
        self.diversity_sampling_strategy = self.configs_model['diversity_sampling']

        # Load dataset parameters
        self.test_patch_dir = self.configs_test['test_patch_dir']
        self.test_patch_size = self.configs_test['test_patch_size']
        self.test_case = self.configs_test['test_case_name']
        self.adata_path = self.configs_test['adata_file']
        
        # Prepare data for prediction            
        self.dataset, self.dataloader = self._prepare_data()
        
        # Initialize model
        self._initialize_model()
    
    def _load_config(self):
        """
        Load the test configuration from a user-provided file.

        Returns
        -------
        dict
            Dictionary of test configuration parameters.
        """
        return config_utils.read_config_from_user_file(self.config_file_path)

    def _load_model_config(self):
        """
        Load the model configuration from the run save directory.

        Returns
        -------
        dict
            Dictionary of model configuration parameters.
        """
        run_save_dir = os.path.join(os.path.dirname(self.configs_test['model_savepath']))
        return config_utils.read_config_from_file(os.path.join(run_save_dir, "config.json"))

    def _load_mapping(self):
        """
        Load the label mapping from a JSON file.

        Returns
        -------
        dict
            Mapping from class indices (int) to class labels (str).
    """
        mapping_path = os.path.join(self.run_save_dir, "labels_mapping.json")
        with open(mapping_path, "r") as file:
            mapping = json.load(file)
        return {int(key): value for key, value in mapping.items()}

    def _prepare_data(self):
        """
        Prepare the test dataset and DataLoader.

        Handles both unimodal and multimodal models.

        Returns
        -------
        dataset : torch.utils.data.Dataset
            The prepared test dataset.
        dataloader : torch.utils.data.DataLoader
            DataLoader for the test dataset.
        """
        print('***************************************************************************')
        
        print(self.configs_model['pretraind_model'])
        if 'multimodal' in self.configs_model['pretraind_model']:
            # print in yellow
            print('\033[93m' + 'Using multimodal model' + '\033[0m')
            
            dataset = PrecomputedImageDataset_Multimodal(self.configs_test['hdf5_file'], self.configs_test['adata_file'], 'valid', transform=None, return_image_path=True, normalise_image_features=self.configs_model['normalise_image_features'], normalise_ge_features=self.configs_model['normalise_genomic_features'])
            
        else:    
            # print in yellow
            print('\033[93m' + 'Using unimodal model' + '\033[0m')
            dataset = PrecomputedImageDataset(self.configs_test['hdf5_file'], 'valid', transform=None, return_image_path=True, normalise_image_features=self.configs_model.get('normalise_image_features', False))
        
        
        
        if 'study_wise_normalisation' in self.configs_test and self.configs_test['study_wise_normalisation']:
            dataset.preprocessing_transfer_learning(source_dataset='whole_study')
            
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        return dataset, dataloader

    def _initialize_model(self):
        """
        Initialize and load the model for inference.

        Loads the model architecture and weights, moves the model to the selected device,
        and sets it to evaluation mode.
        """
        print("Initializing AL_model...")
        self.AL_model = model_utils.init_model(self.configs_model, num_classes=self.number_of_classes)
        print("Creating model...")
        self.AL_model.create_model()
        print("Loading model weights from..."+self.model_path)
        source_state_dict = torch.load(self.model_path)
        self.AL_model.load_state_dict(source_state_dict, strict=True)
        print("Moving model to device...")
        self.AL_model = self.AL_model.to(self.device)
        print("Setting model to evaluation mode...")
        self.AL_model.eval()

    def make_predictions(self):
        """
        Generate predictions and uncertainty scores for the test dataset.

        Iterates through the test DataLoader, applies the model, computes predictions,
        and calculates uncertainty metrics for each sample.

        Returns
        -------
        predictions_df : pandas.DataFrame
            DataFrame containing barcodes, model predictions, and uncertainty scores.
        """
        predictions_df = pd.DataFrame(columns=['Barcode', colname_in_pred, 'Uncertainty'])
        barcodes, predictions, uncertainties = [], [], []
        # print in yellow path to model
        print('\033[93m' + f'Path to model: {self.model_path}' + '\033[0m')
        # Loop through dataloader to generate predictions
        for batch_data in tqdm(self.dataloader):
            for key in batch_data.keys():
                if key != 'Barcode':
                    batch_data[key] = batch_data[key].to(self.device)
                    
            batch_outputs = self.AL_model(batch_data)
            batch_outputs = batch_outputs.detach()
            _, batch_preds = torch.max(batch_outputs, 1)
            
            batch_image_names = batch_data['Barcode']
            # Collect predictions and barcodes
            predictions.extend(batch_preds.cpu().numpy())
            barcodes.extend([re.findall(r'[TGCA]+\-[0-9]', image)[0] for image in batch_image_names])
            
            # Calculate uncertainty scores
            uncertainty_scores = calculate_uncertainty(batch_outputs, self.select_uncertainty, self.number_of_classes)
            uncertainties.extend(uncertainty_scores)
        
        predictions_df['Barcode'] = barcodes
        predictions_df[colname_in_pred] = predictions
        predictions_df['Uncertainty'] = uncertainties
        
        return predictions_df

    def make_predictions_return_features(self):
        """
        Generate predictions, uncertainty scores, and extract intermediate features for the test dataset.

        This method processes the test DataLoader, applies the model in feature-extraction mode,
        and collects barcodes, predictions, uncertainty scores, and feature vectors for each sample.

        Returns
        -------
        predictions_df : pandas.DataFrame
            DataFrame containing barcodes, model predictions, uncertainty scores, and extracted features.
        """
        predictions_df = pd.DataFrame(columns=['Barcode', colname_in_pred, 'Uncertainty', 'Features'])	
        barcodes, predictions, uncertainties = [], [], []
        features_list = []
        # print in yellow path to model
        print('\033[93m' + f'Path to model: {self.model_path}' + '\033[0m')
        # Loop through dataloader to generate predictions
        for batch_data in tqdm(self.dataloader):
            for key in batch_data.keys():
                if key != 'Barcode':
                    batch_data[key] = batch_data[key].to(self.device)
                    
            batch_outputs, features = self.AL_model(batch_data, return_features=True)
            batch_outputs = batch_outputs.detach()
            features = features.detach()

            _, batch_preds = torch.max(batch_outputs, 1)
            
            batch_image_names = batch_data['Barcode']
            # Collect predictions and barcodes
            predictions.extend(batch_preds.cpu().numpy())
            barcodes.extend([re.findall(r'[TGCA]+\-[0-9]', image)[0] for image in batch_image_names])
            features_list.extend(features.cpu().numpy())
            
            # Calculate uncertainty scores
            uncertainty_scores = calculate_uncertainty(batch_outputs, self.select_uncertainty, self.number_of_classes)
            uncertainties.extend(uncertainty_scores)
        
        predictions_df['Barcode'] = barcodes
        predictions_df[colname_in_pred] = predictions
        predictions_df['Uncertainty'] = uncertainties
        predictions_df['Features'] = features_list
        
        return predictions_df

    def save_predictions(self, predictions_df, save_name='Model_predictions'):
        """
        Save model predictions to disk, optionally replacing predictions for training spots
        with ground truth annotations in active learning runs.

        If the test case matches the training dataset, predictions for training spots are
        replaced with ground truth (already provided annotations by the expert). Otherwise, predictions are saved as-is
        (e.g., for transfer learning scenarios).

        Parameters
        ----------
        predictions_df : pandas.DataFrame
            DataFrame containing model predictions, indexed by barcode.
        save_name : str, optional
            Base name for the saved file. Defaults to 'Model_predictions'.

        """
        # save raw predictions
        predictions_df.set_index('Barcode', inplace=True)
        
        predictions_df[colname_in_pred] = predictions_df[colname_in_pred].map(self.mapping)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # If it active learning run
        if self.configs_test['test_case_name'] == self.configs_model['dataset']:
            # save predictions where training spots are replaced with ground truth
            training_spots = pd.read_csv(self.configs_model['loupe_csv_file'], index_col=0)
            # common index - need to account for cases where SpaceRanger is reruned after inital annotations, some spots miht be excluded in this case compare to original annotations
            common_indices = training_spots.index.intersection(predictions_df.index)
            
            missing_in_predictions = training_spots.index.difference(predictions_df.index)
            
            
            # replace training spots with ground truth in predictions
            predictions_df.loc[common_indices, colname_in_pred] = training_spots.loc[common_indices, gt_column_name]
        
            
            filename = f'{save_name}_{self.test_case}_{timestamp}.csv'
            # save only Barcode and Model Prediction
            predictions_df[[colname_in_pred]].to_csv(os.path.join(self.run_save_dir, filename), index=True)
            
            # Step 3: Handle Missing Indices on Both Sides
            if not missing_in_predictions.empty:
                logging.warning(
                    f"The following {len(missing_in_predictions)} indices from training_spots are not present in predictions_df and were skipped:\n"
                    f"{missing_in_predictions.tolist()}"
                )
        else:
            # if it is transfer learning run, save predictions as is
            filename = f'{save_name}_{self.test_case}__no_trainingset_predictions_corrected_{timestamp}.csv'
            # save only Barcode and Model Prediction
            predictions_df[[colname_in_pred]].to_csv(os.path.join(self.run_save_dir, filename), index=True)                     
           
    def run_diversity_sampling(self, top_sorted_predictions):
        """
        Perform diversity sampling on the model predictions.

        Calls the configured diversity sampling strategy to select a diverse subset of barcodes
        for annotation, using HDF5 features as specified.

        Parameters
        ----------
        top_sorted_predictions : pandas.DataFrame or np.ndarray
            Top predictions sorted by uncertainty or other criteria.

        Returns
        -------
        barcodes_selected : list
            List of selected barcodes after diversity sampling.
        """
        # Run diversity sampling
        barcodes_selected = diversity_sampling(
            self.diversity_sampling_strategy,
            top_sorted_predictions,
            self.configs_test['hdf5_file'],
            self.configs_model['total_number_of_samples_for_all_clusters'],
            self.device,
            self.adata_path,
            use_pca=self.configs_model['use_PCA']
        )
        
        return barcodes_selected

    def obtain_predictions(self):
        """
        Orchestrate the prediction, selection, and annotation workflow for active learning.

        This method generates model predictions (and optionally features), saves the results,
        and selects new samples for annotation based on uncertainty and diversity sampling strategies.
        It also ensures that samples already annotated or present in test/validation sets are excluded.

        Returns
        -------
        int or None
            Returns 1 if in fully supervised mode; otherwise, returns None.
        """
        
        if self.configs_test['diversity_sampling'] == "computed_features":
            predictions_df = self.make_predictions_return_features()
        else:
            predictions_df = self.make_predictions()
            
        self.save_predictions(predictions_df)
        
        
        if self.mode == 'fully_supervised':
            return 1
        
        if self.configs_test['test_case_name'] == self.configs_model['dataset']:
            # Remove barcodes that are already annotated and in the test set
            
            training_set_barcodes = pd.read_csv(self.configs_model['loupe_csv_file'], index_col=0).index.values
            predictions_df = predictions_df[~predictions_df.index.isin(training_set_barcodes)]
            
            # check if test_data exists
            if os.path.exists(self.configs_model['test_data']):        
                test_set_barcodes = pd.read_csv(self.configs_model['test_data'], index_col=0).index.values
                predictions_df = predictions_df[~predictions_df.index.isin(test_set_barcodes)]
                
            # check if valid_data exists
            if os.path.exists(self.configs_model['valid_data']):
                valid_set_barcodes = pd.read_csv(self.configs_model['valid_data'], index_col=0).index.values
                predictions_df = predictions_df[~predictions_df.index.isin(valid_set_barcodes)]
        
        # sort predictions by uncertainty
        sorted_predictions = predictions_df.sort_values(by='Uncertainty', ascending=False)
        
        # check if percentage_of_samples_to_select is 'exp_decay
        if self.configs_model.get('percentage_of_samples_to_select', 0.05) == 'exp_decay':        
            decay_rate = 0.7
            base_percentage = 0.05
            init_percentage = 0.15 #0.3 # 0.5
            steps = self.configs_model['num_active_learning_runs']-1
            percentage = max(base_percentage, init_percentage * math.exp(-decay_rate*steps))
            # exponential decay
            print("Total number of active learning runs: ", self.configs_model['num_active_learning_runs'])
            print("Steps: ", steps)
            print("Percentage of samples to select: ", percentage)
            print("Len of sorted predictions: ", len(sorted_predictions))
            
            
            num_selected_samples = int(len(sorted_predictions) * percentage)        
        elif self.configs_model.get('percentage_of_samples_to_select', 0.05) == 'fixed_value':
            num_selected_samples = self.configs_model['total_number_of_samples_for_all_clusters'] *  self.configs_model['train_num_classes']
            
        else:
            num_selected_samples = int(len(sorted_predictions) * self.configs_model.get('percentage_of_samples_to_select', 0.05))
        
        
        print(f'Selected {num_selected_samples} samples with highest uncertainty')
        
        
        if self.select_uncertainty == 'none':
            top_sorted_predictions = sorted_predictions.sample(num_selected_samples)
        else:
            top_sorted_predictions = sorted_predictions.head(num_selected_samples)
        
        # Run diversity sampling
        barcodes_selected = self.run_diversity_sampling(top_sorted_predictions)
        
        # Save selected samples for annotation
        self._save_selected_samples(barcodes_selected)

    def _save_selected_samples(self, barcodes_selected):
        """
        Save the selected barcodes for annotation, ensuring no duplicates and correct sample count.

        For active learning runs, combines new samples with current annotations and checks for
        consistency. For transfer learning, saves new samples to a separate file.

        Parameters
        ----------
        barcodes_selected : list
            List of barcodes selected for annotation.

        """
        todo_array = ["help" for _ in barcodes_selected]
        data = {'Barcode': barcodes_selected, gt_column_name: todo_array}
        df = pd.DataFrame(data).set_index('Barcode')
        
        if self.configs_test['test_case_name'] == self.configs_model['dataset']:
            current_annotations = pd.read_csv(self.configs_model['loupe_csv_file'], index_col=0)
            df = pd.concat([current_annotations, df])
            
            number_of_help = len(df[df[gt_column_name] == 'help'])
            total_number_of_samples_for_all_clusters = self.configs_model['total_number_of_samples_for_all_clusters']
            
            assert number_of_help == total_number_of_samples_for_all_clusters, f'Number of Pathologist Annotations that are  "help" should be {total_number_of_samples_for_all_clusters} but is {number_of_help}'

            # check if there are duplicates
            duplicates = df[df.index.duplicated()]
            assert duplicates.empty, f'There are duplicates in df: {duplicates}'

            # save df to to_annotate.csvs
            if not self.debug:
                df.to_csv(self.configs_model['loupe_csv_file'], index=True)

        else:
            if not self.debug:
                filename = 'Uncertain_samples_' + self.test_case + '.csv'
                df.to_csv(os.path.join(self.run_save_dir, filename), index=True)
        print('***************************************************************************')
        print(f"Time elapsed: {time.time() - start_time}", flush=True)
        print('***************************************************************************')

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run ActiveVisium with config and optional WandB logging.")
    parser.add_argument("config_file_path", help="Path to the configuration file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (default: disabled).")
    
    args = parser.parse_args()

    config_file_path = args.config_file_path
    debug = args.debug
    
    app_manager = ApplicationManager(config_file_path, debug=debug)
    app_manager.obtain_predictions()
    
    
    if not debug:
        os._exit(0)
