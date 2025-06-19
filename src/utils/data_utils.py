import pandas as pd
import os
from sklearn.model_selection import train_test_split
from data.Dataset import PrecomputedLoupeDataset, PrecomputedLoupeDataset_Multimodal
import torch
import multiprocessing
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    # Check if the batch has only one sample, if so, return an empty list
    # This is important for batch normalization layers
    if len(batch) == 1:
        return {}  # This will skip batches with a single sample
    return torch.utils.data.dataloader.default_collate(batch)


def create_data_loaders(my_configs, train_spots, valid_spots, external_validation_spots, mapping, return_barcode=False):

    num_workers = 0 #min(4, multiprocessing.cpu_count())
    
    train_dataset = PrecomputedLoupeDataset(
        train_spots, 
        my_configs.get('hdf5_file'), 
        mapping=mapping, 
        mode='train', 
        transform=None, 
        return_barcode=return_barcode, 
        normalise_image_features=my_configs.get('normalise_image_features', False)
    )
    valid_dataset = PrecomputedLoupeDataset(
        valid_spots, 
        my_configs.get('hdf5_file'), 
        mapping=mapping, 
        mode='valid', 
        transform=None, 
        return_barcode=return_barcode, 
        normalise_image_features=my_configs.get('normalise_image_features', False)
    )

    # check if study_wise_normalisation is in the config file and true  
    if 'study_wise_normalisation' in my_configs and my_configs['study_wise_normalisation']:
        train_dataset.preprocessing_transfer_learning(source_dataset='whole_study')
        valid_dataset.preprocessing_transfer_learning(source_dataset='whole_study')
    
    
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=my_configs['batch_size'], shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=my_configs['batch_size'], shuffle=True, num_workers=num_workers,
                                                    collate_fn=custom_collate_fn)
    
    
    print('Train dataset size: ', len(train_dataset))
    print('Valid dataset size: ', len(valid_dataset))

    if external_validation_spots is not None:
        external_validation_dataset = PrecomputedLoupeDataset(
            loupe_dataframe=external_validation_spots, 
            hdf5_file=my_configs.get('hdf5_file'), 
            mapping=mapping, 
            transform=None, 
            mode='valid', 
            return_barcode=False, 
            normalise_image_features=my_configs.get('normalise_image_features', False)
        )
        
        if 'study_wise_normalisation' in my_configs and my_configs['study_wise_normalisation']:
            external_validation_dataset.preprocessing_transfer_learning(source_dataset='whole_study')
        
        print('External validation dataset size: ', len(external_validation_dataset))
        
        external_validation_dataloader = torch.utils.data.DataLoader(external_validation_dataset, batch_size=my_configs['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
            
        dataset_sizes = {'train':len(train_dataset), 'valid':len(valid_dataset), 'external_validation':len(external_validation_dataset)}
        dataloaders = {'train':dataloader_train, 'valid':dataloader_valid, 'external_validation':external_validation_dataloader}
    else:
        dataset_sizes = {'train':len(train_dataset), 'valid':len(valid_dataset)}
        dataloaders = {'train':dataloader_train, 'valid':dataloader_valid}    

    return dataloaders, dataset_sizes

def create_data_loaders_multimodal(my_configs, train_spots, valid_spots, external_validation_spots, mapping, return_barcode=False):

    
    num_workers = 0 #min(4, multiprocessing.cpu_count())
    
    train_dataset = PrecomputedLoupeDataset_Multimodal(train_spots, my_configs['hdf5_file'],my_configs['adata_file'], mapping=mapping, mode='train', transform=None, return_barcode=return_barcode, normalise_image_features=my_configs['normalise_image_features'], normalise_ge_features=my_configs['normalise_genomic_features'])
    
    valid_dataset = PrecomputedLoupeDataset_Multimodal(valid_spots, my_configs['hdf5_file'],my_configs['adata_file'], mapping=mapping, mode='valid', transform=None, return_barcode=return_barcode, normalise_image_features=my_configs['normalise_image_features'], normalise_ge_features=my_configs['normalise_genomic_features'])
    
    if 'study_wise_normalisation' in my_configs and my_configs['study_wise_normalisation']:
        train_dataset.preprocessing_transfer_learning(source_dataset='whole_study')
        valid_dataset.preprocessing_transfer_learning(source_dataset='whole_study')
    
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=my_configs['batch_size'], shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    dataloader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=my_configs['batch_size'], shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
    
    
    print('Train dataset size: ', len(train_dataset))
    print('Valid dataset size: ', len(valid_dataset))

    if external_validation_spots is not None:
        external_validation_dataset = PrecomputedLoupeDataset_Multimodal(loupe_dataframe=external_validation_spots, hdf5_file=my_configs['hdf5_file'],adata_file=my_configs['adata_file'], mapping=mapping, transform=None, mode='valid',return_barcode=False, normalise_image_features=my_configs['normalise_image_features'], normalise_ge_features=my_configs['normalise_genomic_features'])
        
        if 'study_wise_normalisation' in my_configs and my_configs['study_wise_normalisation']:
            external_validation_dataset.preprocessing_transfer_learning(source_dataset='whole_study')
        
        print('External validation dataset size: ', len(external_validation_dataset))
        
        external_validation_dataloader = torch.utils.data.DataLoader(external_validation_dataset, batch_size=my_configs['batch_size'], shuffle=False, collate_fn=custom_collate_fn)
            
        dataset_sizes = {'train':len(train_dataset), 'valid':len(valid_dataset), 'external_validation':len(external_validation_dataset)}
        dataloaders = {'train':dataloader_train, 'valid':dataloader_valid, 'external_validation':external_validation_dataloader}
    else:
        dataset_sizes = {'train':len(train_dataset), 'valid':len(valid_dataset)}
        dataloaders = {'train':dataloader_train, 'valid':dataloader_valid}    

    return dataloaders, dataset_sizes
    

def load_data(my_configs, run = None):

    train_spots = None
    valid_spots = None
    weights = None
    mapping = None

    annotations, weights, mapping = load_Loupe_annotations(
        my_configs['loupe_csv_file'],
        patch_dir=my_configs['patch_dir'],
        calculate_class_weights=my_configs['calculate_class_weights']
    )
    print("Loaded annotations from: ", my_configs['loupe_csv_file'])
    print(annotations.head())
    # calculate numbers of each annotation in annotations
    annotation_counts = annotations['Pathologist Annotations'].value_counts()
    # if all counts are bigger then 1
    print("Annotation counts:")
    print(annotation_counts)
    
    # Split the data into training and validation sets
    if (annotation_counts > 1).all() and my_configs['valid_proportion']>0:
        train_spots, valid_spots = split_annotations_train_val(annotations, valid_proportion=my_configs['valid_proportion'], seed=my_configs['random_seed'])
    else:
        train_spots = valid_spots = annotations
        if my_configs['valid_proportion']==0:
            print("Valid proportion is set to 0, so all training data will be used for training")
        else:
            print("Don't have enough annotations per class, so all training data will be used for training")
            
    if run is not None:
        sum_of_annotations = annotation_counts.sum()
        annotation_counts = annotation_counts.to_dict()
        run.summary['total_annotations'] = sum_of_annotations
        for key in annotation_counts.keys():
            run.summary[key] = annotation_counts[key]
            
    # print counts of annotations in train and valid
    print('**************************************************************')
    print("Train annotations:")
    print(train_spots['Pathologist Annotations'].value_counts())
    print("Valid annotations:")
    print(valid_spots['Pathologist Annotations'].value_counts())
    print('**************************************************************')
    return train_spots, valid_spots, weights, mapping   

def load_Loupe_annotations(csv_file_path, patch_dir=None, calculate_class_weights=False, ranom_seed=99):
    """
    Load annotations from a CSV file and filter spots with annotations. Ensures that each barcode has a corresponding patch.
    
    Parameters:
        csv_file (str): Path to the CSV file containing annotations.
        patch_dir (str or None): Path to the directory containing patch files (default: None).
        calculate_class_weights (bool or str): Whether to calculate class weights. If 'ICF', inverse class frequencies will be used (default: False).
        
    Returns:
        loupe_csv_file (pd.DataFrame): DataFrame containing filtered annotations.
        weights (pd.Series or None): Series containing calculated class weights if applicable, otherwise None.
    """
    # Read the CSV file
    csv_file = pd.read_csv(csv_file_path)
    print("Loaded annotations from: ", csv_file_path)
    print(csv_file)
    
    # Get column names from the CSV file
    column_names = list(csv_file.columns.values)
    print("Column names in the CSV file: ", column_names)
    
    # Select rows with non-null annotations (spots that contain annotations)
    loupe_csv_file = csv_file[~csv_file[column_names[1]].isna()]
    if patch_dir is not None:
        # Get list of image files in the patch directory
        print("Checking for patches in directory: ", patch_dir)
        image_files = os.listdir(patch_dir)
        
        # Extract barcodes from image filenames
        image_barcodes = [(filename.split('_')[1]).split('.')[0] for filename in image_files if filename.startswith('patch_')]
        
        if len(image_barcodes) > 0:
            # Filter annotations to ensure each barcode has a corresponding patch
            loupe_csv_file = loupe_csv_file[loupe_csv_file[column_names[0]].isin(image_barcodes)]

    weights = None
    if calculate_class_weights == 'ICF':
        # Calculate class weights based on inverse class frequencies
        class_frequencies = loupe_csv_file[column_names[1]].value_counts(normalize=True)
        inverse_class_frequencies = 1 / class_frequencies
        # Normalize the weights so that they sum up to 1
        weights = inverse_class_frequencies / sum(inverse_class_frequencies)
    elif calculate_class_weights == 'ICF_not_normalized':
        # Calculate class weights based on inverse class frequencies
        class_frequencies = loupe_csv_file[column_names[1]].value_counts(normalize=True)
        inverse_class_frequencies = 1 / class_frequencies
        # Normalize the weights so that they sum up to 1
        weights = inverse_class_frequencies
    elif calculate_class_weights == 'standard_weights':
        # standard class weights
        class_frequencies = loupe_csv_file[column_names[1]].value_counts(normalize=True)
        inverse_class_frequencies = 1 / class_frequencies
        total_number_of_samples = len(loupe_csv_file)
        total_number_of_classes = len(inverse_class_frequencies)
        weights = total_number_of_samples/(total_number_of_classes*class_frequencies)
    elif calculate_class_weights == 'standard_weights_v2':
        # standard class weights
        class_frequencies = loupe_csv_file[column_names[1]].value_counts(normalize=False)
        inverse_class_frequencies = 1 / class_frequencies
        total_number_of_samples = len(loupe_csv_file)
        total_number_of_classes = len(inverse_class_frequencies)
        weights = total_number_of_samples/(total_number_of_classes*class_frequencies)
            

    column_names = list(csv_file.columns.values)

    # Define label-int mapping
    unique_labels = loupe_csv_file[column_names[1]].unique()
    mapping={}
    for (i,label) in enumerate(unique_labels):
        mapping[i]=label

    return loupe_csv_file, weights, mapping

def split_annotations_train_val(annotations, valid_proportion=0.2, seed=99):
    """
    Splits a DataFrame of annotations into training and validation sets while maintaining class distribution.
    
    Parameters:
        annotations (pd.DataFrame): The input DataFrame containing annotations.
        valid_proportion (float): The proportion of data to allocate for the validation set (default: 0.2).
        seed (int): Seed for random number generation (default: 99).
    
    Returns:
        train_df (pd.DataFrame): DataFrame containing training data annotations.
        valid_df (pd.DataFrame): DataFrame containing validation data annotations.
    """
    # Shuffle the input DataFrame
    shuffled_df = annotations.sample(frac=1, random_state=seed)
    
    # Extract column names from the shuffled DataFrame
    column_names = list(shuffled_df.columns.values)
    
    # Group images by class labels
    class_groups = shuffled_df.groupby(column_names[1])
    
    # Initialize empty DataFrames for train and validation sets
    train_df = pd.DataFrame(columns=[column_names[0], column_names[1]])
    valid_df = pd.DataFrame(columns=[column_names[0], column_names[1]])
    
    # Create train-validation split for each class while maintaining class distribution
    for _, group_df in class_groups:
        train_group, valid_group = train_test_split(group_df, test_size=valid_proportion, random_state=seed)
        
        # Append train and validation data for this class to respective DataFrames
        train_df = pd.concat([train_df, train_group])
        valid_df = pd.concat([valid_df, valid_group])
    
    # Reset the index of the DataFrames
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    
    return train_df, valid_df
