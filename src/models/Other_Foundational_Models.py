from Abstract_Model import AbstractModel, ResNetTrunk
from torchvision.models.resnet import Bottleneck
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
try:
    from huggingface_hub import login
except ImportError:
    print("Huggingface_hub can't be imported")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class CTransPathBaseModel(AbstractModel):
    def __init__(self, config, num_classes):
        super(CTransPathBaseModel, self).__init__(config, num_classes)
        self.latent_dim = 768
        self.image_size = 224
        self.out_features = self.config.get('num_hidden_units', [])[-1] 
    
    def get_statistics(self):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return mean, std
    
    def create_model(self, create_backbone=False):        
        
        if create_backbone:
            from ctranspath.ctran import ctranspath
            from global_constants import ctranspath_model_path
            self.model_pre_trained = ctranspath()
            self.model_pre_trained.head = nn.Identity()
            td = torch.load(ctranspath_model_path)
            self.model_pre_trained.load_state_dict(td['model'], strict=True)
            self.freeze_pretrained()
            
        else:
            self.model_pre_trained = None
            
            
        self.morph_extractor = self.get_morph_extractor()
        self.classifier = self.get_classifier()
            
class SSL_BaseModel(AbstractModel):
    def __init__(self, config, num_classes):
        super(SSL_BaseModel, self).__init__(config, num_classes)
        self.image_size=224
        self.latent_dim = 384
        self.image_size=224
        self.out_features = self.config.get('num_hidden_units', [])[-1] 

    def get_statistics(self):
        # taken from https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
        mean = [ 0.70322989, 0.53606487, 0.66096631 ]
        std = [ 0.21716536, 0.26081574, 0.20723464 ]

        return mean, std
            
    def create_model(self, create_backbone=False):        
        
        if create_backbone:
            self.latent_dim = 2048
            self.patch_size = 16
            self.embed_dim=384
            self.num_heads=6
            self.vit_num_classes=0    
            self.load_ssl_backbone(self.config['pretraind_model'])
            self.freeze_pretrained()    
        else:
            self.model_pre_trained = None
        
        self.morph_extractor = self.get_morph_extractor()
        self.classifier = self.get_classifier()
        
          
            
    def get_pretrained_url(self, key):
        URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
        model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch"
        }

        pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
        return pretrained_url

    def load_ssl_backbone(self, key, **kwargs):
        from_ssl_repository = True
        if key in ["DINO_p16", "DINO_p8", "BT", "MoCoV2", "SwAV"]:
            pretrained_url = self.get_pretrained_url(key)
    
        if key in ["DINO_p16", "DINO_p8", "pathnet_inhouse"]:
            self.model_pre_trained = VisionTransformer(img_size=self.image_size, patch_size=self.patch_size, embed_dim=self.embed_dim, num_heads=self.num_heads, num_classes=self.vit_num_classes, **kwargs)
            self.latent_dim = self.embed_dim

        elif key in ["BT", "MoCoV2", "SwAV"]:
            self.model_pre_trained = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)

        if from_ssl_repository:
            verbose = self.model_pre_trained.load_state_dict(
                torch.hub.load_state_dict_from_url(pretrained_url, progress=True)
            )
        else:
            raise ValueError(f"Unsupported SSL model key: {key}")

        print(verbose)

class ResNetImageNet(AbstractModel):
    def __init__(self, config, num_classes):
        super(ResNetImageNet, self).__init__(config, num_classes)
        self.latent_dim = 2048
        self.image_size = 224
        self.out_features = self.config.get('num_hidden_units', [])[-1] 

    def get_statistics(self):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return mean, std
    
    def create_model(self, create_backbone=False):        
        
        if create_backbone:
            self.model_pre_trained = resnet50(weights='IMAGENET1K_V1')
            self.model_pre_trained = torch.nn.Sequential(*(list(self.model_pre_trained.children())[:-1]))
            self.freeze_pretrained()
                
        else:
            self.model_pre_trained = None
        
        
        self.morph_extractor = self.get_morph_extractor()
        self.classifier = self.get_classifier()
    
class ProvGigaPath(AbstractModel):
    
    def __init__(self, config, num_classes):
        super(ProvGigaPath, self).__init__(config, num_classes)
        self.latent_dim = 1536
        self.image_size = 224
        self.out_features = self.config.get('num_hidden_units', [])[-1] 
        self.create_model()
    
    def get_statistics(self):
        # taken from https://github.com/prov-gigapath/prov-gigapath
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        return mean, std
    
    
    def create_model(self, create_backbone=False):        
        
        if create_backbone:
            self.model_pre_trained = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            self.freeze_pretrained()
            
        else:
            self.model_pre_trained = None
        
        
        self.morph_extractor = self.get_morph_extractor()
        self.classifier = self.get_classifier()

