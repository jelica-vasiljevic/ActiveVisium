def extract_base_name(config_file, fully_supervised=True):
    replace_dict = {
        'unimodal_UNI_none_random_norm_im_0_norm_ge_0': 'Unimodal (Random)',
        'unimodal_UNI_Least_confidence_sampling_computed_features_norm_im_0_norm_ge_0': 'Unimodal (Active Learning)',
        
        'multimodal_UNI_multimodal_none_random_norm_im_0_norm_ge_0': 'Multimodal (Random)',
        'multimodal_UNI_multimodal_Least_confidence_sampling_computed_features_norm_im_0_norm_ge_0': 'Multimodal (Active Learning)',      
        "Fully_supervised_UNI":   "Fully supervised (unimodal)",
        "Fully_supervised_UNI_multimodal": "Fully supervised (multimodal)",
    }
        
    name = replace_dict[config_file]
        
    return name

def get_title(dataset):
    if dataset == 'Sample_SN048_A121573_Rep1':
        return 'Human Colorectal Cancer (FF) \n (Sample_SN048_A121573_Rep1)'
    elif dataset == 'human_kidney':
        return 'Human Healthy  Kidney (FFPE)'
    elif dataset == 'mouse_kidney':
        return 'Mouse Healthy Kidney (FFPE)'
    elif dataset == 'breast':
        return 'Human Breast Cancer (FFPE)'
