"""
Defines the base architecture for ActiveVisium framework. Models will inherit from this class.

It includes:
- An AbstractModel class for shared logic across different backbone variants.
- A ResNetTrunk class for using ResNet without the final classification layer.
"""

import torch.nn as nn
from torchvision.models.resnet import  ResNet
from torchsummary import summary


class AbstractModel(nn.Module):
    def __init__(self, config, num_classes):
        super(AbstractModel, self).__init__()
        self.config = config
        self.num_classes = num_classes

        self.model_pre_trained = None
        self.model_classifier = None
        self.model = None
        self.latent_dim = None
        self.image_size = None

        self.optimiser = None
        self.criterion = None
        
    def freeze_pretrained(self):
        for param in self.model_pre_trained.parameters(): # type: ignore
            param.requires_grad = False

    def get_morph_extractor(self):
        layers = []
        layers.append(nn.Flatten())
        last_feature_size = self.latent_dim
        assert self.latent_dim is not None, "Latent dimension is not set"
        
        for i, hidden_units in enumerate(self.config['num_hidden_units']):
            layers.append(nn.Linear(in_features=last_feature_size, out_features=hidden_units)) # type: ignore
            layers.append(nn.LeakyReLU(negative_slope=0.02))
            if self.config['dropout']:
                layers.append(nn.Dropout(self.config['dropout']))
            last_feature_size = hidden_units

        
        return nn.Sequential(*layers)

    def get_classifier(self):
        return nn.Sequential(
            nn.Linear(in_features=self.out_features, out_features=self.num_classes)
        )
        
    
    
    def forward(self, x, return_features=False, return_sep_features=False):
        # Check input keys
        if 'image' not in x:
            raise ValueError("Input dictionary must contain key 'image'")
        
        morph_features = self.morph_extractor(x['image'])
        logits = self.classifier(morph_features)
        
        if return_features:
            if return_sep_features:
                return logits,morph_features, morph_features, None
            else:
                return logits, morph_features
        else:
            return logits
    

    def print_model_summary(self):
        print(self.morph_extractor)
        print(self.classifier)
 
class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        return x

