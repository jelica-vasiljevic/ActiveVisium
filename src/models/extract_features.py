import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
import torch
import h5py

# Import custom modules
#import models.model_utils as model_utils
import model_utils
from utils import config_utils
from data.Dataset import ImageDataset
from data import Preprocessing

from global_constants import HUGGINGFACE_TOKEN

# Set environment variable for Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_TOKEN


def make_representation(save_path, AL_model, train_loader, valid_loader, device, n_runs=10):
    """
    Generate and save image representations using a pre-trained model.

    Parameters:
    - save_path (str): Path to save the HDF5 file.
    - AL_model (object): The active learning model where feature extractor is (pre-trained) foundational model.
    - train_loader (trainloader): trainloader for the images.
    - valid_loader (validloader): validloader for the images.
    - device (torch.device): Device to run the model on (CPU or GPU).
    - n_runs (int): Number of runs (representations) per image. Default is 10.
    """
    try:
        # Open the HDF5 file in append mode
        with h5py.File(save_path, 'a') as hdf:
            # ------------------------------
            # Process the train_loader (with augmented representations)
            print("Processing augmented representations for training set")
            for run in range(n_runs):
                print(f"Processing augmentation {run + 1}/{n_runs}")
                
                for i, (images, image_paths) in enumerate(tqdm(train_loader, desc=f"Processing augmentations for run {run + 1}")):
                    
                    # Move images to device
                    images = images.to(device)
                    
                    # Get representations from the pre-trained model
                    augmented_features = AL_model.model_pre_trained(images)
                    
                    # Store the representations for each image path
                    for j, path in enumerate(image_paths):
                        sanitized_path = path.replace('/', '_')  # Sanitize path for HDF5 dataset names
                        
                        # Create or access the group for the barcode under 'train'
                        img_group = hdf.require_group(f'train/{sanitized_path}')
                        
                        # Save augmented features as 'aug1', 'aug2', etc.
                        img_group.create_dataset(f'aug{run + 1}', data=augmented_features[j].cpu().detach().numpy())

            # ------------------------------
            # Process the valid_loader (with a single representation)
            print("Processing representations for validation set")
            for i, (images, image_paths) in enumerate(tqdm(valid_loader, desc="Processing validation set")):
                
                # Move images to device
                images = images.to(device)
                
                # Get representations from the pre-trained model
                validation_features = AL_model.model_pre_trained(images)
                
                # Store the representations for each image path
                for j, path in enumerate(image_paths):
                    sanitized_path = path.replace('/', '_')  # Sanitize path for HDF5 dataset names
                    
                    # Create or access the group for the barcode under 'valid'
                    img_group = hdf.require_group(f'valid/{sanitized_path}')
                    
                    # Save the validation features as 'rep'
                    img_group.create_dataset('rep', data=validation_features[j].cpu().detach().numpy())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    """
    Main function to execute the script.

    Parameters:
    - config_file_path (str): Path to the configuration file.
    - num_repeats (int, optional): Number of repetitions for data augmentation. Default is 10.
    """
    config_file_path = sys.argv[1]
    
    # Check if the number of repetitions for data augmentation is provided
    if len(sys.argv) > 2:
        num_repeats = int(sys.argv[2])
    else:
        num_repeats = 10
    
    # Read configuration from the user-provided file
    my_config = config_utils.read_config_from_user_file(config_file_path)
    
    print(f"Building model with config: {my_config['pretraind_model']}")
    
    # Build model
    num_classes = 1
    AL_model = model_utils.init_model(my_config, num_classes=num_classes)
    AL_model.create_model(create_backbone=True)
    
    # Load patches
    patch_dir = my_config['patch_dir']
    # Get list of image files that end with "png" in the patch directory
    image_files = [filename for filename in os.listdir(patch_dir) if filename.endswith('png')]
    print(f"Found {len(image_files)} image files in {patch_dir}")
    
    if not image_files:
        print("No image files found. Exiting.")
        sys.exit(1)
    
    # Prepare dataset
    mean, std = AL_model.get_statistics()
    img_transform_train, img_transform_valid = Preprocessing.get_image_transformation(mean=mean, std=std, image_size=AL_model.image_size)
    
    train_dataset = ImageDataset(image_files, img_transform_train, patch_dir, return_image_path=True)
    valid_dataset = ImageDataset(image_files, img_transform_valid, patch_dir, return_image_path=True)
    
    # Create a DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AL_model.model_pre_trained = AL_model.model_pre_trained.to(device)
    
    # Ensure save directory exists
    save_path = os.path.join(os.path.dirname(patch_dir), f'{my_config["pretraind_model"]}_{num_repeats}_representations.h5')
    # assert that the save directory exists
    assert os.path.exists(os.path.dirname(save_path)), f"Directory {os.path.dirname(save_path)} does not exist."
    
    # Generate and save representations
    make_representation(save_path, AL_model, train_loader, valid_loader, device, n_runs=num_repeats)
