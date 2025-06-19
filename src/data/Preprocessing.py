from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

def get_image_transformation(mean, std, image_size):
    
    # ************************************************
    # set Transformations
    # ************************************************
    img_transform_train = None
    img_transform_valid = None

    train_transforms = []
    valid_transforms = []

    
    train_transforms.append(VerticalFlip(p=.5))
    train_transforms.append(HorizontalFlip(p=.5))
    train_transforms.append(Rotate(p=0.5))
    train_transforms.append(Blur(p=.5, blur_limit=5))
    train_transforms.append(GaussNoise(p=.5, var_limit=5.0))
    train_transforms.append(Resize(image_size,image_size))
    valid_transforms.append(Resize(image_size,image_size))
    
    train_transforms.append(RandomGamma(p=0.5, gamma_limit=(40, 250))) # albumentations expects gamma values in different range then other libraries, experimtally these values seem to work well
    train_transforms.append(Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True))
    train_transforms.append(ToTensor())
    
    valid_transforms.append(Normalize(mean=mean, std=std, max_pixel_value=255.0, always_apply=True))
    valid_transforms.append(ToTensor())
    
    img_transform_train = Compose(train_transforms)
    img_transform_valid = Compose(valid_transforms)

   
    return img_transform_train, img_transform_valid
        
