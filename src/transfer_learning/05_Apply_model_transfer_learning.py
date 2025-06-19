# Import standard libraries
import os
# add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                

import pandas as pd
import re
import json
import time
import sys
from datetime import datetime
import torch.nn.functional as F
import torch
from global_constants import gt_column_name,colname_in_pred
# Import third-party libraries
from tqdm import tqdm

# Import local modules
from utils import config_utils
from models import model_utils
from data.Dataset import PrecomputedImageDataset, PrecomputedImageDataset_Multimodal
from data import Preprocessing
from active_learning_methods.uncertainty import calculate_uncertainty
from active_learning_methods.diversity import diversity_sampling

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

class ApplicationManager:
    def __init__(self, config_file_path, debug=False):
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
        if os.path.samefile(self.configs_model['loupe_csv_file'], self.configs_model['train_data']):
            print("This is fully supervised setting, so no new samples are added to the training set.")           
            self.mode = 'fully_supervised'
     
               
        # Set model parameters
        self.AL_model = None
        self.number_of_classes = self.configs_model['train_num_classes']
        self.model_path = self.configs_model['model_savepath']
        self.autoencoder_model_path = self.configs_model.get('autoencoder_model_path', None)
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
        return config_utils.read_config_from_user_file(self.config_file_path)

    def _load_model_config(self):
        run_save_dir = os.path.join(os.path.dirname(self.configs_test['model_savepath']))
        return config_utils.read_config_from_file(os.path.join(run_save_dir, "config.json"))

    def _load_mapping(self):
        mapping_path = os.path.join(self.run_save_dir, "labels_mapping.json")
        with open(mapping_path, "r") as file:
            mapping = json.load(file)
        return {int(key): value for key, value in mapping.items()}

    def _prepare_data(self):
        # check if model is multimodal
        print('***************************************************************************')
        
        print(self.configs_model['pretraind_model'])
        if 'multimodal' in self.configs_model['pretraind_model']:
            # print in yellow
            print('\033[93m' + 'Using multimodal model' + '\033[0m')
            
            dataset = PrecomputedImageDataset_Multimodal(self.configs_test['hdf5_file'], self.configs_test['adata_file'], 'valid', transform=None, return_image_path=True, normalise_image_features=self.configs_model['normalise_image_features'], normalise_ge_features=self.configs_model['normalise_genomic_features'])
        else:    
            # print in yellow
            print('\033[93m' + 'Using unimodal model' + '\033[0m')
            dataset = PrecomputedImageDataset(self.configs_test['hdf5_file'], 'valid', transform=None, return_image_path=True, normalise_image_features=self.configs_model['normalise_image_features'])
        
        # print in yellow
        print('\033[93m' + 'This is transfer learning setting, we need to use the same preprocessing as in the training set' + '\033[0m')
        print('***************************************************************************')
        
        print('Test dataset:'+ self.configs_test['test_case_name'])
        print('Original dataset:'+ self.configs_model['dataset'])
        
        dataset.preprocessing_transfer_learning(source_dataset=self.configs_model['dataset'])
        
        
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        return dataset, dataloader

    def _initialize_model(self):
        print("Initializing AL_model...")
        self.AL_model = model_utils.init_model(self.configs_model, num_classes=self.number_of_classes)
        print("Creating model...")
        self.AL_model.create_model()
        print("Loading model weights...")
        source_state_dict = torch.load(self.model_path)
        self.AL_model.load_state_dict(source_state_dict, strict=True)
        print("Moving model to device...")
        self.AL_model = self.AL_model.to(self.device)
        print("Setting model to evaluation mode...")
        self.AL_model.eval()

    def make_predictions(self):
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
            uncertainty_scores = calculate_uncertainty(batch_outputs, self.select_uncertainty, self.number_of_classes, None)
            uncertainties.extend(uncertainty_scores)
        
        predictions_df['Barcode'] = barcodes
        predictions_df[colname_in_pred] = predictions
        predictions_df['Uncertainty'] = uncertainties
        
        return predictions_df

    def save_predictions(self, predictions_df, save_name='Model_predictions'):
        # save raw predictions
        predictions_df.set_index('Barcode', inplace=True)
        
        predictions_df[colname_in_pred] = predictions_df[colname_in_pred].map(self.mapping)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{save_name}_{self.test_case}__corrected_adata_no_trainingset_predictions_corrected_{timestamp}.csv'
        # save only Barcode and Model Prediction
        predictions_df[[colname_in_pred]].to_csv(os.path.join(self.run_save_dir, filename), index=True)
        
        
        # only if dataset is the same as test_case_name
        if self.configs_test['test_case_name'] == self.configs_model['dataset']:
            # save predictions where training spots are replaced with ground truth
            training_spots = pd.read_csv(self.configs_model['loupe_csv_file'], index_col=0)
            # replace training spots with ground truth in predictions
            predictions_df.loc[training_spots.index, colname_in_pred] = training_spots[gt_column_name]
            
            filename = f'{save_name}_{self.test_case}_{timestamp}.csv'
            # save only Barcode and Model Prediction
            predictions_df[[colname_in_pred]].to_csv(os.path.join(self.run_save_dir, filename), index=True)
           
    def run_diversity_sampling(self, top_sorted_predictions):

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
        predictions_df = self.make_predictions()
        self.save_predictions(predictions_df)
        
        
        if self.mode == 'fully_supervised':
            return 1
        
        # if self.configs_test['test_case_name'] == self.configs_model['dataset']:
        #     # Remove barcodes that are already annotated and in the test set
            
        #     training_set_barcodes = pd.read_csv(self.configs_model['loupe_csv_file'], index_col=0).index.values
        #     predictions_df = predictions_df[~predictions_df.index.isin(training_set_barcodes)]
            
        #     # check if test_data exists
        #     if os.path.exists(self.configs_model['test_data']):        
        #         test_set_barcodes = pd.read_csv(self.configs_model['test_data'], index_col=0).index.values
        #         predictions_df = predictions_df[~predictions_df.index.isin(test_set_barcodes)]
                
        #     # check if valid_data exists
        #     if os.path.exists(self.configs_model['valid_data']):
        #         valid_set_barcodes = pd.read_csv(self.configs_model['valid_data'], index_col=0).index.values
        #         predictions_df = predictions_df[~predictions_df.index.isin(valid_set_barcodes)]
        
        # # sort predictions by uncertainty
        # sorted_predictions = predictions_df.sort_values(by='Uncertainty', ascending=False)
        # num_selected_samples = int(len(sorted_predictions) * 0.05)
        # print(f'Selected {num_selected_samples} samples with highest uncertainty')
        
        
        # if self.select_uncertainty == 'none':
        #     top_sorted_predictions = sorted_predictions.sample(num_selected_samples)
        # else:
        #     top_sorted_predictions = sorted_predictions.head(num_selected_samples)
        
        # # Run diversity sampling
        # barcodes_selected = self.run_diversity_sampling(top_sorted_predictions)
        
        # # Save selected samples for annotation
        # self._save_selected_samples(barcodes_selected)

    def _save_selected_samples(self, barcodes_selected):
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
    config_file_path = sys.argv[1]
    debug = len(sys.argv) > 2 and sys.argv[2] == "True"
    
    app_manager = ApplicationManager(config_file_path, debug=debug)
    app_manager.obtain_predictions()
    
    if not debug:
        os._exit(0)
