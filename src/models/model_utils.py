import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Models import *
from Other_Foundational_Models import *

def init_model(my_configs, num_classes):
    """
    Initializes and returns a model based on the configuration provided.

    Parameters
    ----------
    my_configs : dict
        A configuration dictionary containing at least the key `'pretraind_model'` (str),
        which specifies the name of the pretrained model to load. It may also contain
        other model-specific hyperparameters or settings.
    
    num_classes : int
        The number of output classes for the classification task. Passed to the model's constructor.

    Returns
    -------
    model : torch.nn.Module
        An initialized PyTorch model instance corresponding to the specified pretrained architecture.

    Supported Values for `my_configs['pretraind_model']`
    ------------------------------------------------------
    - `'ctranspath'` :
        Loads a CTransPath transformer-based model.

    - `'BT'`, `'MoCoV2'`, `'SwAV'`, `'DINO_p16'` :
        Loads a self-supervised learning (SSL) base model using one of the supported SSL pretraining methods.

    - `'ResNet50ImageNet'` :
        Loads a ResNet-50 architecture pretrained on ImageNet.

    - `'prov_gigapath'` :
        Loads a custom ProvGigaPath model (likely domain-specific, e.g. pathology).


    - `'UNI_multimodal'` :
        Loads a UNI multimodal foundational model (e.g., for vision + transcriptomics integration).

    - `'UNI'` :
        Loads a base version of the UNI foundational model.

    Notes
    -----
    - This function expects all model classes to be imported from `Models` or `Other_Foundational_Models`.
    - If the `pretraind_model` key is missing or set to an unsupported value, the function will fail silently (returns `None`).
      Notes
    -----
    - All model classes must be properly imported from `Models` or `Other_Foundational_Models`.
    - If the `pretraind_model` key is missing or set to an unsupported value, the function will fail silently (i.e., `model` will not be defined).
    - ⚠️ **If new model architectures are added**, this function must be manually updated to include the corresponding `elif` clause and import the appropriate class.

    """

    if my_configs['pretraind_model'] == 'ctranspath':
        model = CTransPathBaseModel(config=my_configs, num_classes=num_classes)

    elif my_configs['pretraind_model'] in ['BT', 'MoCoV2', 'SwAV','DINO_p16']:
        model =  SSL_BaseModel(config=my_configs, num_classes=num_classes)
        
    elif my_configs['pretraind_model'] == 'ResNet50ImageNet':
        model = ResNetImageNet(my_configs,num_classes) 

    elif my_configs['pretraind_model'] == 'prov_gigapath':
        model = ProvGigaPath(config=my_configs, num_classes=num_classes)
        
    elif my_configs['pretraind_model'] == 'UNI_multimodal':
        model = UNI_Multimodal_HVG(config=my_configs, num_classes=num_classes)

    elif my_configs['pretraind_model'] == 'UNI':
        model = UNI(config=my_configs, num_classes=num_classes)
    
    return model
    
        