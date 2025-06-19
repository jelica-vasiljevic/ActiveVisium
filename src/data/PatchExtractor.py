import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import pandas as pd
import random
import os
import cv2
from PIL import ImageFilter
import argparse
import skimage
from skimage.filters import threshold_mean
import matplotlib.pyplot as plt
import sys
import json
from PIL import Image
from tqdm import tqdm


class PatchExtractor():
    '''
    Class for patch extraction from WSI. 
    '''

    def __init__(self, imagePath, patch_size=None, visium_data=True, max_number_of_patches=20, image_scaling_factor=1):

        self.imagePath = imagePath
        self.patch_size = patch_size
        self.use_visium_data = visium_data
        self.max_nb_patches = max_number_of_patches
        self.image_scaling_factor = image_scaling_factor # represent the scaling factor between WSI used for path extraction and WSI used for SpaceRanger output

        try:
            self.imageSvs = openslide.OpenSlide(self.imagePath)
            print(f'The number of levels in the slide {self.imageSvs.level_count}')
            print(f'Dimensions {self.imageSvs.dimensions}')
            print(f'Level_Dimensions {self.imageSvs.level_dimensions}')
            print(f'Level Downsample {self.imageSvs.level_downsamples}')

            self.full_w,self.full_h = self.imageSvs.level_dimensions[0]
            print(f" Image width and height at full resolution is {self.full_w}, {self.full_h}")
        except:
            print(f"Error when loading image {self.imagePath}. It might not be in proper OpenSlide format ")
            return            
            
        
        if visium_data:
            print("Extract patrches using SpaceRanger output")
            print("Scale factor between WSI used for patch extraction and WSI used for SpaceRanger output is " + str(self.image_scaling_factor))
                  
            self.extraction_data_path = os.path.dirname(self.imagePath)
            self.save_path = os.path.join(self.extraction_data_path,'extracted_patches')
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            # Loading tissue positions from SpaceRanger output files

            tissue_postions_path = os.path.join(self.extraction_data_path,"spatial","tissue_positions.csv")
            
            if os.path.exists(tissue_postions_path):
                pos_path = tissue_postions_path
            else:
                tissue_postions_path = os.path.join(self.extraction_data_path,"spatial","tissue_positions_list.csv")
                if os.path.exists(tissue_postions_path):
                    pos_path = tissue_postions_path
                else:
                    raise FileNotFoundError(f"Tissue positions list not found in directory {os.path.join(self.extraction_data_path,'spatial')}")  # type: ignore
             
            self.pos = pd.read_csv(pos_path)
            if 'in_tissue' not in list(self.pos.columns.values):
                self.pos = pd.read_csv(pos_path, names=["barcode", "in_tissue", "array_y", "array_x", "pxl_row_in_fullres", "pxl_col_in_fullres"])

            self.pos_tissue = self.pos[self.pos['in_tissue']==1]
            self.spot_coordinates = self.pos_tissue[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy()
            self.spot_coordinates_with_barcode = self.pos_tissue[['barcode','pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy()

            # Loading Scaling Facotrs
            scaling_factors_path = os.path.join(self.extraction_data_path,"spatial","scalefactors_json.json")
            if os.path.exists(scaling_factors_path):
                with open(scaling_factors_path) as json_file:
                    self.scaling_factors = json.load(json_file)
            else:
                raise FileNotFoundError(f"File scalefactors_json seems to be missing in directory {os.path.join(self.extraction_data_path,'spatial')}") # type: ignore
        else:
            raise NotImplementedError("Patch extraction for non-Visium data is not implemented yet. Please set visium_data=True to use SpaceRanger output.")
    
            
        
    def extract_patches_based_on_spot_coordinates(self, save_folder_name='exctracted_patches_based_on_spots', only_tissue_under_spot = True):
        nb_extracted = 0
        if self.patch_size is None:
            #https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/spatial
            patch_size = int(np.round(self.scaling_factors['spot_diameter_fullres'])) # The number of pixels that span the diameter of a theoretical 65µm spot in the original, full-resolution image. 
            print(f"Original Patch size is {patch_size}")
            patch_shift = int(np.round(patch_size // 2))
            print(f"Original Patch shift is {patch_shift}")

            scale_factor = self.image_scaling_factor
            print(f"Scale factor is {scale_factor}")

            patch_size = int(np.round(patch_size // scale_factor)) 
            patch_shift = int(np.round(patch_shift // scale_factor))
            self.patch_size = patch_size
            self.patch_shift = patch_shift
        else:
            self.patch_shift = int(self.patch_size // 2)

        # print in red color patch size and patch shift

        print(f"\033[0;31;47m Patch size is {self.patch_size}\033[00m")
        print(f"\033[0;31;47m Patch shift is {self.patch_shift}\033[00m")
    
        
        save_path = os.path.join(self.save_path,save_folder_name,f"spot_patches_zoom_{self.image_scaling_factor}_ps_{self.patch_size}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print("Files saved at" + save_path)
        for i in tqdm(range(0,min(self.max_nb_patches,self.spot_coordinates_with_barcode.shape[0]))):
            barcode,row,col = self.spot_coordinates_with_barcode[i] 
            # row/cols are coordinates with respect to the image provided during running SpaceRanger. If this image is downsampled from the original WSI, and original WSI is used for patch extraction, we need to scale these numbers
            startX = int(row/self.image_scaling_factor)
            startY = int(col/self.image_scaling_factor)

            # always reading patch from the original WSI, so level=0
            patch = self.imageSvs.read_region((startX - self.patch_shift, startY - self.patch_shift), 0, (self.patch_size, self.patch_size))
            
            patch.save(f"{save_path}/patch_{barcode}.png")
            nb_extracted = nb_extracted + 1
        return nb_extracted

    
    def visualise_spots(self):
    
        im = Image.open(self.extraction_data_path + '/spatial/tissue_hires_image.png')
        if self.use_visium_data:
            spots_row  = self.spot_coordinates[:,0] * self.scaling_factors['tissue_hires_scalef']
            spots_cols = self.spot_coordinates[:,1] * self.scaling_factors['tissue_hires_scalef']
        else:
            spots_row  = self.spot_coordinates_with_barcode[:,1] * self.manual_scaling_factor
            spots_cols = self.spot_coordinates_with_barcode[:,2] * self.manual_scaling_factor
        
        plt.imshow(im)
        plt.scatter(x=spots_row, y=spots_cols, s=2)
        plt.show()
        plt.savefig(os.path.join(self.save_path,'debug_spots.png'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract patches.')

    parser.add_argument('-img', '--imgPath', type=str, help='Path to WSI image, shoud be in root folder when used with Visum data')
    parser.add_argument('-scalingFactor', '--scalingFactor', default=1, type=float, help='Scaling factor between WSI used for patch extraction and WSI used for SpaceRanger output')
    parser.add_argument('-ps', '--patchSize', default=None, type=int, help='Size of patches to be extracted')
    parser.add_argument('-maxNB', '--maxNB', default=sys.maxsize, type=int, help='Max number of patches to be extracted')
    parser.add_argument('-spaceRangerOutput', '--spaceRangerOutput', default=1, type=int, help='Are tissue possitions and scaling factors correct with respect to provided WSI?')
    

    args = parser.parse_args()
    print(args)
    extractor = PatchExtractor(args.imgPath, args.patchSize, max_number_of_patches = args.maxNB, visium_data=args.spaceRangerOutput, image_scaling_factor=args.scalingFactor)
    
    extractor.visualise_spots()

    num_patches = extractor.extract_patches_based_on_spot_coordinates()
    
    print(f"Sucesfully extracted {num_patches} patches")


