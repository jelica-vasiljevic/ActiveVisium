
import json
import sys
import os
from pathlib import Path

def write_config_to_file(config, file_path):
    with open(file_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

def read_config_from_user_file(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    
    config_path = Path(file_path).resolve()
    config_dir = (config_path.parent).parent
    
    # print(f"\n[DEBUG] Config path: {config_path}")
    # print(f"[DEBUG] Config dir: {config_dir}")
    # print(f"[DEBUG] base_path raw: {config.get('base_path')}")

    base_path_raw = config.get("base_path", "")
    base_path = Path(base_path_raw)
    if not base_path.is_absolute():
        base_path = (config_dir / base_path).resolve(strict=True)
    
    # print(f"[DEBUG] base_path resolved: {base_path}")

    config["base_path"] = str(base_path)


    # Replace placeholders with values
    config['original_annotations_test_only'] = config['original_annotations_test_only'].format(base_path=config['base_path'], dataset=config['dataset'])
    
    config['initial_spot_selection'] = config['initial_spot_selection'].format(pretraind_model=config['pretraind_model'])
    if 'multimodal' in config['pretraind_model']:
        # remove multimodal from initial_spot_selection
        config['initial_spot_selection'] = config['initial_spot_selection'].replace('_multimodal', '')
    
    
    
    config['loupe_csv_file'] = config['loupe_csv_file'].format(
    base_path=config['base_path'],
    dataset=config['dataset'],
    pretraind_model=config['pretraind_model'],
    uncertainty_metric=config['uncertainty_metric'],
    calculate_class_weights=config['calculate_class_weights'],
    use_stain_augmentation=config['use_stain_augmentation'],
    diversity_sampling=config['diversity_sampling'],
    repetition=config['repetition'],
    initial_spot_selection=config['initial_spot_selection'],
    normalise_image_features=config.get('normalise_image_features', ''),
    normalise_genomic_features=config.get('normalise_genomic_features', ''), )
    
    config['adata_file'] = config['adata_file'].format(base_path=config['base_path'], dataset=config['dataset'],test_case_name=config['test_case_name'])
    config['patch_dir'] = config['patch_dir'].format(base_path=config['base_path'], dataset=config['dataset'], train_patch_size=config['train_patch_size'])
    config["test_patch_dir"]= config["test_patch_dir"].format(base_path=config['base_path'],test_case_name=config['test_case_name'],test_patch_size = config['test_patch_size'])
    
    
    config['model_savepath'] = config['model_savepath'].format(
    base_path=config['base_path'],
    dataset=config['dataset'],
    pretraind_model=config['pretraind_model'],
    uncertainty_metric=config['uncertainty_metric'],
    calculate_class_weights=config['calculate_class_weights'],
    use_stain_augmentation=config['use_stain_augmentation'],
    diversity_sampling=config['diversity_sampling'],
    repetition=config['repetition'],
    initial_spot_selection=config['initial_spot_selection'],
    normalise_image_features=config.get('normalise_image_features', ''),
    normalise_genomic_features=config.get('normalise_genomic_features', ''), )



    config['test_data']= config['test_data'].format(base_path=config['base_path'],dataset=config['dataset'])
    config['train_data']= config['train_data'].format(base_path=config['base_path'],dataset=config['dataset'])
    config['valid_data']= config['valid_data'].format(base_path=config['base_path'],dataset=config['dataset'])
    
    

    config['hdf5_file'] = config['hdf5_file'].format(patch_dir=config['patch_dir'], pretraind_model=config['pretraind_model'], test_patch_dir=config['test_patch_dir'])
    
    config['test_patch_dir']= config['test_patch_dir'].format(base_path=config['base_path'],test_case_name=config['test_case_name'],test_patch_size = config['test_patch_size'])

    config['project']= config['project'].format(uncertainty_metric=config['uncertainty_metric'])
    config['tag'] = [tag.format(pretraind_model=config['pretraind_model'], 
                                diversity_sampling=config['diversity_sampling'], 
                                calculate_class_weights=config['calculate_class_weights'], 
                                use_stain_augmentation=config['use_stain_augmentation']) 
                    for tag in config['tag']]
    
    return config


def read_config_from_file(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config



