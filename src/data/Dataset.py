import os
import torch
import pandas as pd
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from PIL import Image
import scanpy as sc
import re
import h5py
import random
from global_constants import whole_study_base_path, whole_study_samples, img_extraction_path, gene_extraction_path

def global_calculate_statistics_image_features(hdf5_file, source_dataset=None):
    # Take base name of the HDF5 file
    basename_hdf_file = os.path.basename(hdf5_file).split('.')[0]
    
    if source_dataset is None:
        dataset = hdf5_file.split('/')[5]
        print(dataset)
        print(hdf5_file)
        # print in yellow
        print('\033[93m' + f'Calculating IMAGE statistics for {dataset} dataset' + '\033[0m')
        # Read statistics from {dataset}_statistics.h5 if it exists
        filename = f'{dataset}_statistics_img_features_{basename_hdf_file}.h5'
        
        hdf5_files = [hdf5_file]
    elif source_dataset =='whole_study':
        print('\033[93m' + f'Calculating IMAGE statistics for WHOLE study' + '\033[0m')
        hdf5_files = []
        for sample in whole_study_samples:
            hdf5_files.append(f"{whole_study_base_path}/{sample}/{img_extraction_path}")
        
        filename = f'colorectal_whole_study_statistics_img_features.h5'
    
    else:
        dataset = source_dataset
        filename = f'{dataset}_statistics_img_features_{basename_hdf_file}.h5'
        assert os.path.exists(filename), f"Statistics file {filename} does not exist"
            
    
    if os.path.exists(filename):
        with h5py.File(filename, 'r') as f:
            mean_array = f['Mean'][:]
            std_array = f['Std'][:]
        
        return mean_array, std_array
    
    # Read only valid group and calculate mean and std
    sum_data = None
    sum_data_sq = None
    count = 0
    
    for hdf5_file in hdf5_files:
        # If not, calculate statistics
        img_features = h5py.File(hdf5_file, 'r')
        
        for barcode in img_features['valid']:
            group = img_features['valid'][barcode]
            features = group['rep'][:]
            data = np.array(features, dtype=np.float32)
            if sum_data is None:
                sum_data = np.zeros_like(data)
                sum_data_sq = np.zeros_like(data)
            sum_data += data
            sum_data_sq += data ** 2
            count += 1
    # Calculate mean
    mean = sum_data / count
    
    # Calculate variance
    variance = (sum_data_sq / count) - (mean ** 2)
    
    # Calculate standard deviation
    std = np.sqrt(variance)
    
    # Save mean and std to HDF5 file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('Mean', data=mean)
        f.create_dataset('Std', data=std)
    
    return mean, std

def global_calculate_statistics_gene_expression(adata_file, source_dataset=None):
    
    
    if source_dataset == 'whole_study':
        print('\033[93m' + f'Calculating GENE statistics for WHOLE study' + '\033[0m')
        adata = sc.read_h5ad(f"{whole_study_base_path}/{whole_study_samples[0]}/{gene_extraction_path}")
        for sample in whole_study_samples[1:]:
            adata = adata.concatenate(sc.read_h5ad(f"{whole_study_base_path}/{sample}/{gene_extraction_path}"))
        
        filename = f'colorectal_whole_study_statistics_ge_features.h5'
    
    else:
        # Take base name of the HDF5 file
        basename_adata_file = os.path.basename(adata_file).split('.')[0]
        if source_dataset is None:
            
            dataset = adata_file.split('/')[5]
            
            print('\033[93m' + f'Calculating GENE statistics for {dataset} dataset' + '\033[0m')
            
            filename = f'{dataset}_statistics_ge_features_{basename_adata_file}.h5'
            
        else:
            dataset = source_dataset
            filename = f'{dataset}_statistics_ge_features_{basename_adata_file}.h5'
            assert os.path.exists(filename), f"Statistics file {filename} does not exist"
    
    if os.path.exists(filename):
        with h5py.File(filename, 'r') as f:
            mean_array = f['Mean'][:]
            std_array = f['Std'][:]
        return mean_array, std_array
    
    # if not, calculate statistics    
    adata = sc.read_h5ad(adata_file)
    if 'X_pca_harmony' not in adata.obsm.keys():
        data = adata.obsm['X_pca']
    else:
        data = adata.obsm['X_pca_harmony']
    
    # Calculate mean, it is numpy array
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Save mean and std to HDF5 file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('Mean', data=mean)
        f.create_dataset('Std', data=std)
    
    return mean, std
    
class PrecomputedImageDataset(Dataset):
    """
    Custom dataset class for loading precomputed features from an HDF5 file.

    Parameters:
        hdf5_file (str): Path to the HDF5 file containing precomputed features.
        mode (str): Either 'train' or 'valid', specifying which group to read from.
        transform (callable, optional): Optional transform to be applied on a sample.
        return_image_path (bool, optional): If True, returns the barcode along with the features.
    """
    def __init__(self, hdf5_file, mode, transform=None, return_image_path=False, normalise_image_features=False):
        self.hdf5_file = hdf5_file  # HDF5 file containing precomputed features.
        self.return_image_path = return_image_path
        self.transform = transform  # Optional transform to apply on features.
        self.mode = mode  # Indicates whether to read from 'train' or 'valid'
        
        self.normalise_image_features = normalise_image_features
        
        # Open the HDF5 file once for efficient loading
        self.hdf = h5py.File(self.hdf5_file, 'r')
        self.mode_group = self.hdf[self.mode]
        # Get the list of barcodes (keys) in the specified mode
        self.barcodes = list(self.mode_group.keys())
        
        self.img_mean, self.img_std  = 0.0, 1.0
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file)

    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.barcodes)

    def __getitem__(self, idx):
        """Returns a precomputed feature representation and its corresponding barcode."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the barcode corresponding to the index
        barcode = self.barcodes[idx]
        group = self.mode_group[barcode]
        
        # Handle loading differently depending on whether it's training or validation
        if self.mode == 'train':
            # Get the list of augmentations available
            augmentations = [key for key in group.keys() if key.startswith('aug')]
            # Randomly select one of the augmentations
            aug_key = random.choice(augmentations)
            features = group[aug_key][:]
        else:
            # For validation, simply load the 'rep' representation
            features = group['rep'][:]

        # Apply any transforms (optional)
        if self.transform:
            features = self.transform(features)
        
        # Normalize the features
        features = (features - self.img_mean) / self.img_std
            

        if self.return_image_path:
            return {'image': torch.tensor(features, dtype=torch.float32), 'Barcode': barcode}
        else:
            return {'image': torch.tensor(features, dtype=torch.float32)}

    def __del__(self):
        """Ensure the HDF5 file is closed properly when the dataset is deleted."""
        if hasattr(self, 'hdf') and self.hdf:
            self.hdf.close()

    def preprocessing_transfer_learning(self, source_dataset):
        # print in green
        print('\033[92m' + 'Refining image statistics for transfer learning' + '\033[0m')
        print('Source dataset:', source_dataset)
        
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file, source_dataset)
                
class PrecomputedLoupeDataset(Dataset):
    """
    Custom dataset class for loading Loupe annotations and associated precomputed features.

    Parameters:
        loupe_dataframe (pd.DataFrame): DataFrame containing barcodes and annotations.
        hdf5_file (str): Path to the HDF5 file containing precomputed features.
        mapping (dict): Dictionary mapping labels to numerical indices.
        mode (str): Either 'train' or 'valid', specifying which group to read from.
    """
    def __init__(self, loupe_dataframe, hdf5_file, mapping, mode, transform=None, return_barcode=False, normalise_image_features=False):
        self.loupe_csv_file = loupe_dataframe  # DataFrame containing barcodes and annotations.
        self.hdf5_file = hdf5_file  # HDF5 file containing precomputed features.
        self.transform = transform  # Optional transform to apply on features.
        self.mapping = mapping  # Mapping from class names to numerical indices
        self.mode = mode  # Indicates whether to read from 'train' or 'valid'
        self.return_barcode = return_barcode  # If True, returns the barcode along with the features
        
        self.normalise_image_features = normalise_image_features
        
        # Open the HDF5 file once for efficient loading
        self.hdf = h5py.File(self.hdf5_file, 'r')

        # Replace original labels in the DataFrame with numerical indices
        self.loupe_csv_file_replaced = self.loupe_csv_file.copy()
        unique_labels = self.loupe_csv_file[self.loupe_csv_file.columns[1]].unique()
        for i, _ in enumerate(unique_labels):    
            self.loupe_csv_file_replaced = self.loupe_csv_file_replaced.replace(self.mapping[i], i)
        
        self.img_mean, self.img_std  = 0.0, 1.0
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file)            
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.loupe_csv_file)
    

    def __getitem__(self, idx):
        """Returns a precomputed feature representation and its corresponding label."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get barcode associated with the current index
        barcode_orig = self.loupe_csv_file_replaced.iloc[idx, 0].replace('/', '_')  # Sanitize the barcode
        barcode = "patch_"+barcode_orig+".png"

        # Access the group for the barcode in the specified group ('train' or 'valid')
        group = self.hdf[f'{self.mode}/{barcode}']
        
        # Handle loading differently depending on whether it's training or validation
        if self.mode == 'train':
            # Get the number of augmentations available
            n_runs = len([key for key in group.keys() if key.startswith('aug')])
            # Randomly select one of the augmentations
            run_idx = random.randint(1, n_runs)
            features = group[f'aug{run_idx}'][:]
        else:
            # For validation, simply load the 'rep' representation
            features = group['rep'][:]

        # Get the label associated with the current index
        label = self.loupe_csv_file_replaced.iloc[idx, 1]

        # Apply any transforms (optional, though likely unnecessary for features)
        if self.transform:
            features = self.transform(features)
            
        # Normalize the features
        features = (features - self.img_mean) / self.img_std
        

        if self.return_barcode:
            return {'image': torch.tensor(features, dtype=torch.float32), 'labels': torch.tensor(label, dtype=torch.long),'Barcode': barcode_orig}
        else:
            return {'image': torch.tensor(features, dtype=torch.float32), 'labels': torch.tensor(label, dtype=torch.long)}

    def __del__(self):
        """Ensure the HDF5 file is closed properly when the dataset is deleted."""
        self.hdf.close()
        
    def preprocessing_transfer_learning(self, source_dataset):
        # print in green
        print('\033[92m' + 'Refining image statistics for transfer learning' + '\033[0m')
        print('Source dataset:', source_dataset)
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file, source_dataset)   

class PrecomputedImageDataset_Multimodal(Dataset):
    """
    Custom dataset class for loading Loupe annotations and associated precomputed features.

    Parameters:
        loupe_dataframe (pd.DataFrame): DataFrame containing barcodes and annotations.
        hdf5_file (str): Path to the HDF5 file containing precomputed features.
        mapping (dict): Dictionary mapping labels to numerical indices.
        mode (str): Either 'train' or 'valid', specifying which group to read from.
    """
    def __init__(self, hdf5_file, adata_file, mode, transform=None, return_image_path=False, normalise_image_features=False, normalise_ge_features=False):
        
        self.hdf5_file = hdf5_file  # HDF5 file containing precomputed features.
        self.adata_file = adata_file # adata file containing gene expression data
        self.transform = transform  # Optional transform to apply on features.
        
        self.normalise_ge_features = normalise_ge_features
        self.normalise_image_features = normalise_image_features
        
        self.mode = mode  # Indicates whether to read from 'train' or 'valid'
        self.return_image_path = return_image_path  # If True, returns the barcode along with the features
        
        # Open the HDF5 file once for efficient loading
        self.hdf = h5py.File(self.hdf5_file, 'r')
        self.mode_group = self.hdf[self.mode]
        self.barcodes = list(self.mode_group.keys())
        
        self.adata = sc.read_h5ad(self.adata_file)
        # select only highly variable genes
        self.adata = self.adata[:, self.adata.var['highly_variable']]

        
        # get barcodes associed with the image files
        self.barcodes = [re.findall(r'[TGCA]+\-[0-9]',barcode)[0] for barcode in self.barcodes]
        
        # filter barcodes that are present in the adata object
        self.barcodes = [barcode for barcode in self.barcodes if barcode in self.adata.obs.index]
        
        self.img_mean, self.img_std  = 0.0, 1.0
        self.ge_mean, self.ge_std = 0.0, 1.0
        
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file)
        if self.normalise_ge_features:
            self.ge_mean, self.ge_std = global_calculate_statistics_gene_expression(self.adata_file)
            

        
        
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.barcodes)

              
    def __getitem__(self, idx):
        """Returns a precomputed feature representation and its corresponding label."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the barcode corresponding to the index
        barcode_orig = self.barcodes[idx]
        barcode = "patch_"+barcode_orig+".png"
        group = self.mode_group[barcode]
        
        # Handle loading differently depending on whether it's training or validation
        if self.mode == 'train':
            # Get the list of augmentations available
            augmentations = [key for key in group.keys() if key.startswith('aug')]
            # Randomly select one of the augmentations
            aug_key = random.choice(augmentations)
            features = group[aug_key][:]
        else:
            # For validation, simply load the 'rep' representation
            features = group['rep'][:]

        # Apply any transforms (optional)
        if self.transform:
            features = self.transform(features)

    
        
        # load gene expression data
        ge_data  = self.adata[self.adata.obs.index == barcode_orig].X.toarray()
        ge_data = torch.tensor(ge_data).float()

        # Apply any transforms (optional, though likely unnecessary for features)
        if self.transform:
            features = self.transform(features)

        
        # Normalize the features
        features = (features - self.img_mean) / self.img_std
        ge_data = (ge_data - self.ge_mean) / self.ge_std
        
        if self.return_image_path:
            return {'image': torch.tensor(features, dtype=torch.float32), 'Barcode': barcode_orig, 'gene_expression': ge_data}
        else:
            return {'image': torch.tensor(features, dtype=torch.float32), 'gene_expression': ge_data}
        
            

    def __del__(self):
        """Ensure the HDF5 file is closed properly when the dataset is deleted."""
        self.hdf.close()
    
    def preprocessing_transfer_learning(self, source_dataset):
        print('\033[92m' + 'Refining image statistics for transfer learning' + '\033[0m')
        print('Source dataset:', source_dataset)
        print("Target data",self.adata_file)
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file, source_dataset)
        if self.normalise_ge_features:
            self.ge_mean, self.ge_std = global_calculate_statistics_gene_expression(self.adata_file, source_dataset)
        
        # reordering adata to include the same genes as in the source dataset
        adata_source = sc.read_h5ad(os.path.join(whole_study_base_path, source_dataset,'adata_harmony_1k_highly_variable.h5ad'))
        hvg_genes = adata_source[:, adata_source.var['highly_variable']].var_names
        
        # print intersection between the two datasets
        print('Number of genes in the source dataset:', len(hvg_genes))
        print('Number of genes in the target dataset:', len(self.adata.var_names))
        print('Number of common genes:', len(set(hvg_genes).intersection(set(self.adata.var_names))))
        
        # set counts to zero for genes that are not in the source dataset
        # Create a boolean mask for genes not in the hvg_genes list
        mask = ~self.adata.var_names.isin(hvg_genes)

        # Set counts to 0 for these genes
        self.adata.X[:, mask] = 0.0

        # put genes in the same order as in the source dataset
        self.adata = self.adata[:, hvg_genes]       
        
class PrecomputedLoupeDataset_Multimodal(Dataset):
    """
    Custom dataset class for loading Loupe annotations and associated precomputed features.

    Parameters:
        loupe_dataframe (pd.DataFrame): DataFrame containing barcodes and annotations.
        hdf5_file (str): Path to the HDF5 file containing precomputed features.
        mapping (dict): Dictionary mapping labels to numerical indices.
        mode (str): Either 'train' or 'valid', specifying which group to read from.
    """
    def __init__(self, loupe_dataframe, hdf5_file, adata_file, mapping, mode, transform=None, return_barcode=False, normalise_image_features=False, normalise_ge_features=False):
        self.loupe_csv_file = loupe_dataframe  # DataFrame containing barcodes and annotations.
        self.hdf5_file = hdf5_file  # HDF5 file containing precomputed features.
        self.adata_file = adata_file # adata file containing gene expression data
        self.transform = transform  # Optional transform to apply on features.
        self.mapping = mapping  # Mapping from class names to numerical indices
        self.mode = mode  # Indicates whether to read from 'train' or 'valid'
        self.return_barcode = return_barcode  # If True, returns the barcode along with the features
        
        self.normalise_ge_features = normalise_ge_features
        self.normalise_image_features = normalise_image_features
        
        # Open the HDF5 file once for efficient loading
        self.hdf = h5py.File(self.hdf5_file, 'r')
        
        self.adata = sc.read_h5ad(self.adata_file)
        # select only highly variable genes
        self.adata = self.adata[:, self.adata.var['highly_variable']]

        # Replace original labels in the DataFrame with numerical indices
        self.loupe_csv_file_replaced = self.loupe_csv_file.copy()
        unique_labels = self.loupe_csv_file[self.loupe_csv_file.columns[1]].unique()
        for i, _ in enumerate(unique_labels):    
            self.loupe_csv_file_replaced = self.loupe_csv_file_replaced.replace(self.mapping[i], i)
        
        # filter barcodes that are present in the adata object
        self.loupe_csv_file_replaced = self.loupe_csv_file_replaced[self.loupe_csv_file_replaced.iloc[:, 0].isin(self.adata.obs.index)]
        
        
        self.img_mean, self.img_std  = 0.0, 1.0
        self.ge_mean, self.ge_std = 0.0, 1.0
        
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file)
        if self.normalise_ge_features:
            self.ge_mean, self.ge_std = global_calculate_statistics_gene_expression(self.adata_file)
            
        

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.loupe_csv_file_replaced)

    def __getitem__(self, idx):
        """Returns a precomputed feature representation and its corresponding label."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get barcode associated with the current index
        barcode_orig = self.loupe_csv_file_replaced.iloc[idx, 0].replace('/', '_')  # Sanitize the barcode
        barcode = "patch_"+barcode_orig+".png"

        # Access the group for the barcode in the specified group ('train' or 'valid')
        group = self.hdf[f'{self.mode}/{barcode}']
        
        # Handle loading differently depending on whether it's training or validation
        if self.mode == 'train':
            # Get the number of augmentations available
            n_runs = len([key for key in group.keys() if key.startswith('aug')])
            # Randomly select one of the augmentations
            run_idx = random.randint(1, n_runs)
            features = group[f'aug{run_idx}'][:]
        else:
            # For validation, simply load the 'rep' representation
            features = group['rep'][:]

        # Get the label associated with the current index
        label = self.loupe_csv_file_replaced.iloc[idx, 1]
        
        # load gene expression data
        ge_data  = self.adata[self.adata.obs.index == barcode_orig].X.toarray()
        # convert to tensor
        ge_data = torch.tensor(ge_data).float()


        # Normalize the features
        features = (features - self.img_mean) / self.img_std
        ge_data = (ge_data - self.ge_mean) / self.ge_std
        
        # Apply any transforms (optional, though likely unnecessary for features)
        if self.transform:
            features = self.transform(features)

        if self.return_barcode:
            return {'image': torch.tensor(features, dtype=torch.float32), 'labels': torch.tensor(label, dtype=torch.long),'Barcode': barcode_orig, 'gene_expression': ge_data}	
        else:
            return {'image': torch.tensor(features, dtype=torch.float32), 'labels': torch.tensor(label, dtype=torch.long), 'gene_expression': ge_data}
        
            

    def __del__(self):
        """Ensure the HDF5 file is closed properly when the dataset is deleted."""
        self.hdf.close()
    
    def preprocessing_transfer_learning(self, source_dataset):
        print('\033[92m' + 'Refining image statistics for transfer learning' + '\033[0m')
        print('Source dataset:', source_dataset)
        if self.normalise_image_features:
            self.img_mean, self.img_std = global_calculate_statistics_image_features(self.hdf5_file, source_dataset)
        if self.normalise_ge_features:
            self.ge_mean, self.ge_std = global_calculate_statistics_gene_expression(self.adata_file, source_dataset)

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for image files.

    Attributes:
        image_files (list): List of image file names.
        transform (albumentations.Compose): Image transformation pipeline.
        pathdir (str): Directory path of the image files.
    """
    def __init__(self, image_files, transform, pathdir, return_image_path=False):
        self.image_files = image_files
        self.transform = transform
        self.pathdir = pathdir
        self.return_image_path = return_image_path

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves an image and applies transformations.

        Args:
            idx (int): Index of the image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        image_path = self.image_files[idx]
        image = cv2.imread(os.path.join(self.pathdir, image_path))
        assert image is not None, f"Image {os.path.join(self.pathdir, image_path)} is None"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transform(image=image)['image']
        if self.return_image_path:
            return image, image_path
        else:
            return image    
        
