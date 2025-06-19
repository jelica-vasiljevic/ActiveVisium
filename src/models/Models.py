
import os
import torch
import torch.nn as nn
import timm
from huggingface_hub import login
from Abstract_Model import AbstractModel

class UNI(AbstractModel):
    def __init__(self, config, num_classes):
        super(UNI, self).__init__(config, num_classes)
        self.latent_dim = 1024
        self.image_size = 224
        self.out_features = self.config.get('num_hidden_units', [])[-1] 
        
    def get_statistics(self):
        # taken from https://github.com/mahmoodlab/UNI
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        return mean, std
    
    
    def create_model(self, create_backbone=False):        
        
        if create_backbone:
            
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token is None:
                raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")
            
            login(token) 
            self.model_pre_trained = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            self.freeze_pretrained()
            
        else:
            self.model_pre_trained = None
        
        
        self.morph_extractor = self.get_morph_extractor()
        self.classifier = self.get_classifier()
        
class UNI_Multimodal_HVG(UNI):
    
    def __init__(self, config, num_classes):
        super(UNI_Multimodal_HVG, self).__init__(config, num_classes)
    
    def get_combination_layer(self, n_features):
        combo = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(in_features=n_features, out_features=self.out_features),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.25)
        )
        return combo
        
    
    def get_gene_extractor(self):
        layers = [
            nn.Linear(in_features=1000, out_features=self.out_features),
            nn.LeakyReLU(negative_slope=0.02),
        ]

        return nn.Sequential(*layers)
                
    def create_model(self):
        self.morph_extractor = self.get_morph_extractor()
        self.gene_extractor = self.get_gene_extractor()
        self.combo_layer = self.get_combination_layer(2*self.out_features)
        self.classifier = self.get_classifier()
        
    def forward(self, x, return_features=False, return_sep_features=False):
        # Check input keys
        if 'image' not in x or 'gene_expression' not in x:
            raise ValueError("Input dictionary must contain keys 'image' and 'gene_expression'")
        
        morph_features = self.morph_extractor(x['image'])
        gene_features = x['gene_expression'].squeeze(1)
        gene_features = self.gene_extractor(gene_features)
        
        combined_features = torch.cat((morph_features, gene_features), dim=1)
        combined_features = self.combo_layer(combined_features)
        
        # add layer to combine features
        
        logits = self.classifier(combined_features)
        
        if return_features:
            if return_sep_features:
                return logits, combined_features, morph_features, gene_features
            else:
                return logits, combined_features
        else:
            return logits

    def print_model_summary(self):
        print("Morph Extractor: ")
        print(self.morph_extractor)
        print("Gene Extractor: ")
        print(self.gene_extractor)
        print("Classifier: ")
        print(self.classifier)
        print("Combo Layer: ")
        print(self.combo_layer)        

