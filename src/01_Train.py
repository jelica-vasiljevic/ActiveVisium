import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
import json
import os
import shutil
import pandas as pd
import time
from datetime import datetime
import sys
import warnings
import argparse
from utils import data_utils, plot_utils
from utils import config_utils
from models import model_utils

from global_constants import max_num_active_learning_runs
# Suppress warnings
warnings.filterwarnings("ignore")

class ExperimentManager:
    """
    Manages the lifecycle of an experiment including configuration, data loading,
    model initialization, training, and logging.
    """
    def __init__(self, config_file_path, use_wandb=False):
        """
        Initialize the ExperimentManager.

        Args:
            config_file_path (str): Path to the user configuration file.
            use_wandb (bool): Whether to enable Weights & Biases logging.
        """
        self.config_file_path = config_file_path
        
        self.use_wandb = use_wandb
        
        self.config_userfile = config_utils.read_config_from_user_file(self.config_file_path)
        
        self.run_save_dir = os.path.dirname(self.config_userfile['model_savepath'])
        os.makedirs(self.run_save_dir, exist_ok=True)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        self.configs = None
        # Prepare experiment directories and configuration
        self.mapping = None
        self.weights = None
        self.dataloaders = None
        self.dataset_sizes = None
        
        
        self._resume_experiment()

    def _initialize_wandb(self):
        """
        Initialize a Weights & Biases (wandb) run for experiment tracking.

        Returns:
            wandb.Run: The initialized wandb run object.
        """

        run_id = wandb.util.generate_id()  # type: ignore
        run =  wandb.init(
            id=run_id,
            resume="allow",
            project=self.configs['project'],
            config=self.configs,
            tags=self.configs['tag']
        )
        
        self.configs['wandb_run_id'] = run_id
        return run
        

    def _resume_experiment(self):
        """
        Resume an experiment from the last saved configuration, or start a new one if none exists.
        Handles incrementing the active learning run count and prepares data.
        """
        print("Resuming experiments")

        config_file  = os.path.join(self.run_save_dir, "config.json")
        
        if not os.path.exists(config_file):
            # there is no config file in run_save_dir
            self.configs = self.config_userfile
            
        else:
            # there is a config file in run_save_dir, read it
            self.configs = config_utils.read_config_from_file(config_file)
            self.mapping = self._load_mapping()
                    
        self.configs['num_active_learning_runs'] = self.configs.get('num_active_learning_runs', 0) + 1
        self.configs['best_valid_loss'] = self.configs.get('best_valid_loss', -1)
        
        self.run = self._initialize_wandb() if self.use_wandb else None
    
        if self.configs['num_active_learning_runs'] >= max_num_active_learning_runs:
            print("Reached maximum number of active learning runs")
            sys.exit(0)
        
        config_utils.write_config_to_file(self.configs, os.path.join(self.run_save_dir, "config.json"))        
        self._preload_data()
        
    def _load_mapping(self):
        """
        Load the label mapping from a JSON file in the run save directory.

        Returns:
            dict: Mapping from class indices to class labels.
        """
        mapping_path = os.path.join(self.run_save_dir, "labels_mapping.json")
        with open(mapping_path, "r") as file:
            mapping = json.load(file)
        return {int(k): v for k, v in mapping.items()}

    def _preload_data(self):
        """
        Ensure that the starting training set exists. If not, copy it from the data directory.
        """
        if not os.path.exists(self.configs['loupe_csv_file']):
            source_path = os.path.join(os.path.dirname(self.configs['original_annotations_test_only']), f"TrainingAnnotations{self.configs['initial_spot_selection']}.csv")
            print(f"Copying loupe_csv_file from: {source_path}")
            shutil.copy(source_path, self.configs['loupe_csv_file'])
            print(f'\033[91m Done! Loupe_csv_file copied to {self.configs["loupe_csv_file"]} \033[0m')
        else:
            print(f'\033[92m Loupe_csv_file already exists: {self.configs["loupe_csv_file"]} \033[0m')

    def load_data(self):
        """
        Load training and validation data, set up label mapping, and create data loaders.

        Returns:
            int: Returns 1 upon successful data loading.
        """
        
        self.train_spots, self.valid_spots, self.weights, mapping_from_data = data_utils.load_data(self.configs, run=self.run)
        
        # check if valid_data path exist
        if 'valid_data' in self.configs and os.path.exists(self.configs['valid_data']):
            external_validation_data = pd.read_csv(self.configs['valid_data'])
        else:
            external_validation_data = None
            
        
        if self.mapping is None:
            self.mapping = mapping_from_data
            self.configs['train_num_classes']=len(self.mapping) #type:ignore
            with open(self.run_save_dir+"/labels_mapping.json", "w") as file:
                json.dump(self.mapping, file)
        
        if self.use_wandb:
            wandb.save(self.run_save_dir+"/labels_mapping.json")  # type: ignore
        
        # Create dataloaders
        if 'multimodal' not in self.configs['pretraind_model']:
            # print in yellow that model is not multimodal
            print("\033[93m Using unimodal model \033[0m")
            self.dataloaders, self.dataset_sizes = data_utils.create_data_loaders(self.configs, self.train_spots, self.valid_spots, external_validation_data,self.mapping)
        else:
            print("\033[93m Using multimodal model \033[0m")
            self.dataloaders, self.dataset_sizes = data_utils.create_data_loaders_multimodal(self.configs, self.train_spots, self.valid_spots, external_validation_data,self.mapping)
        
        return 1
        
    def create_model(self, num_classes, load_pretrained=True):
        """
        Initialize the model architecture for the current experiment.

        If `load_pretrained` is True and a previously saved model checkpoint exists,
        the model weights will be loaded from that checkpoint, allowing for model
        fine-tuning or continued training across active learning runs. If
        `load_pretrained` is set to False, the model will be initialized from
        scratch at the start of each active learning run, which is an approach
        sometimes adopted in the literature to avoid potential bias from earlier
        training stages.

        Args:
            num_classes (int): Number of output classes for the model.
            load_pretrained (bool, optional): Whether to load model weights from a
                previous checkpoint if available. Defaults to True.

        Returns:
            torch.nn.Module: The initialized model, with weights loaded if applicable.
        """       
        AL_model = model_utils.init_model(self.configs, num_classes=num_classes)
        AL_model.create_model()

        if load_pretrained and os.path.exists(self.configs['model_savepath']):
            best_model_path = self.configs['model_savepath']
            AL_model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded model from: {best_model_path}")
        else:
            print("Starting from scratch")
        
        AL_model = AL_model.to(self.device)
        AL_model.print_model_summary()
        return AL_model

    def train_model(self,AL_model):
        """
        Train the provided model using the loaded data, and save the best model based on validation loss.

        Args:
            AL_model (torch.nn.Module): The model to train.
        """        
        criterion, optimizer, scheduler = self._setup_training(AL_model)
        dataloaders = self.dataloaders
        dataset_sizes = self.dataset_sizes
        
        best_model_params_path = self.configs['model_savepath']
        num_epochs = self.configs['epochs']
        device = self.device
        
        if self.use_wandb:
            artifact = wandb.Artifact(name='best_model', type='model')
        
        # check if 'external_validation' is in dataloaders
        if 'external_validation' not in dataloaders:
            phase_names = ['train', 'valid']
            best_loss = -1
            log_loss={'train':[], 'valid':[],'external_validation':[]}
            log_acc={'train':[], 'valid':[],'external_validation':[]}
        else:
            phase_names = ['train', 'external_validation']
            best_loss= self.configs['best_valid_loss']
            log_loss={'train':[], 'external_validation':[]}
            log_acc={'train':[], 'external_validation':[]}
        
        
        # print keys in dataloaders
        print("Keys in dataloaders: ", dataloaders.keys())
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in phase_names:
                print(phase)

                running_loss = 0.0
                running_corrects = 0

                if phase == 'train':
                    AL_model.train()
                else:
                    AL_model.eval()
                
                for  _, data in enumerate(dataloaders[phase]):
                    if not data:  # Checks if the dictionary is empty
                        continue
                    # put all keys from data dictionary to device
                    for key in data.keys():
                        data[key] = data[key].to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == 'train':

                        with torch.set_grad_enabled(True):
                            outputs = AL_model(data)
                            _, preds = torch.max(outputs, 1) #get predicted class
                            loss = criterion(outputs, data['labels'])

                            loss.backward()
                            optimizer.step()
                    else:
                        with torch.set_grad_enabled(False):
                            outputs = AL_model(data)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, data['labels'])

                    # statistics
                    running_loss += loss.item()*data['image'].size(0)
                    running_corrects += (torch.sum(preds == data['labels'].data)).double()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
            
                
                log_loss[phase].append(epoch_loss)
                log_acc[phase].append(epoch_acc.cpu().numpy())
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if (phase!='train'):
                    if ((epoch_loss < best_loss) or best_loss==-1):
                        prev_loss = best_loss
                        best_loss = epoch_loss
                        torch.save(AL_model.state_dict(), best_model_params_path)
                        # if phase == 'external_validation':
                        #     # write best loss in config file
                        self.configs['best_valid_loss'] = best_loss
                        config_utils.write_config_to_file(self.configs, self.run_save_dir+"/config.json")
                                
                        # print in green that best model is changed
                        print("\033[92m Best model is updated! \033[0m")
                        print(f"Best loss updated from {prev_loss} to: {best_loss:.4f}")
                    else:
                        print(f"Best loss is still: {best_loss:.4f}")

                
                if phase == 'train' and scheduler is not None:
                    scheduler.step(log_loss[phase][-1]) #epoch + i / iters) #type:ignore

                if self.run is not None:    
                    # log loss and accuracy to wandb
                    self.run.log({f'{phase}_loss': log_loss[phase][-1]})
                    self.run.log({f'{phase}_acc': log_acc[phase][-1]})
                    
        
        # plot loss and accuracy and save into run_save_dir
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S") 
        plot_utils.plot_loss_acc(log_loss, log_acc, self.run_save_dir, phase_names, current_time)
        if self.use_wandb:
            artifact.add_file(best_model_params_path) # type: ignore           

    def _setup_training(self, AL_model):
        """
        Set up the loss function, optimizer, and learning rate scheduler for training.

        Args:
            AL_model (torch.nn.Module): The model to be trained.

        Returns:
            tuple: (criterion, optimizer, scheduler)
        """
        weights = self._get_class_weights()
        criterion = self._get_loss_function(weights)
        optimizer = self._get_optimizer(AL_model)
        scheduler = self._get_scheduler(optimizer)
        return criterion, optimizer, scheduler

    def _get_class_weights(self):
        """
        Get the class weights for the loss function.

        Returns:
            list or None: The class weights if available, otherwise None.
        """
        return self.weights

    def _get_loss_function(self, weights):
        """
        Create the loss function, optionally using class weights.

        Args:
            weights (list or None): Class weights for imbalanced datasets.

        Returns:
            torch.nn.Module: The loss function.
        """
        if weights is not None:
            # print in RED that class weights are used
            print("\033[91m Class weights are used! \033[0m")
            print("Class weights: ", weights)
            print("*"*50)
            
            class_weights = [weights[self.mapping[i]] for i in self.mapping.keys()]  # type: ignore
            return nn.CrossEntropyLoss(
                #label_smoothing=self.configs['label_smoothing'],
                weight=torch.tensor(class_weights, dtype=torch.float).to(self.device)
            )
        else:
            # print in yellow that class weights are not used
            print("\033[93m Class weights are not used! \033[0m")
            #return nn.CrossEntropyLoss(label_smoothing=self.configs['label_smoothing'])
            return nn.CrossEntropyLoss()

    def _get_optimizer(self, AL_model):
        """
        Initialize the optimizer for training.

        Args:
            AL_model (torch.nn.Module): The model whose parameters will be optimized.

        Returns:
            torch.optim.Optimizer: The optimizer instance.

        Raises:
            ValueError: If the optimizer name is not recognized.
        """
        lr = self.configs['learning_rate']
        optimizer_name = self.configs['oprimiser']
        if optimizer_name == 'SGD':
            return optim.SGD(AL_model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(AL_model.parameters(), lr=lr)
        elif optimizer_name == 'Adam':
            return optim.Adam(AL_model.parameters(), lr=lr)
        elif optimizer_name == 'Adam_L2':
            return optim.Adam(AL_model.parameters(), lr=lr, weight_decay=0.01)
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_scheduler(self, optimizer):
        """
        Initialize the learning rate scheduler. Currently, it only uses ReduceLROnPlateau.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to schedule.

        Returns:
            torch.optim.lr_scheduler.ReduceLROnPlateau: The learning rate scheduler.
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
def main(config_file_path, use_wandb=False):
    """
    Main starting point for running the experiment.

    Args:
        config_file_path (str): Path to the configuration file.
        use_wandb (bool): Whether to enable Weights & Biases logging.
    """
    experiment = ExperimentManager(config_file_path, use_wandb)

    experiment.load_data()
    
    # Load and initialize model
    AL_model = experiment.create_model(len(experiment.mapping), load_pretrained=True)
    
    # Train the model
    experiment.train_model(AL_model)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ActiveVisium with config and optional WandB logging.")
    parser.add_argument("config_file_path", help="Path to the configuration file.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging (default: enabled).")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging.")
    
    args = parser.parse_args()



    config_file_path = args.config_file_path
    use_wandb = args.use_wandb

    # check if global does WANDB_API_KEY exists in environment variables
    if "WANDB_API_KEY" not in os.environ:
        print("WANDB_API_KEY not found in environment variables. Please set it before running the script.")
        use_wandb = False
    
    start_time = time.time()
    main(config_file_path, use_wandb=use_wandb)
    # flush prints and time elapsed
    print("Time elapsed: ", time.time() - start_time, flush=True)
    print("Done!", flush=True)
    sys.exit(0)
