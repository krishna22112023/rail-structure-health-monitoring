import sys
import click
import os
import glob
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

import logging

from pathlib import Path
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root))

logger = logging.getLogger(__name__)

from config import settings
from utils.preprocess_images import PreProcess

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(input_dir,output_dir):

    if not os.path.exists(Path(settings.BASE,output_dir,'processed_skeleton')):
        os.makedirs(Path(settings.BASE,output_dir,'processed_skeleton'))

    if not os.path.exists(Path(settings.BASE,output_dir,'processed_overlay')):
        os.makedirs(Path(settings.BASE,output_dir,'processed_overlay'))

    image_files = (glob.glob(os.path.join(input_dir, "*.jpg")) +
                   glob.glob(os.path.join(input_dir, "*.jpeg")) +
                   glob.glob(os.path.join(input_dir, "*.JPEG")) +
                   glob.glob(os.path.join(input_dir, "*.png")))

    if not image_files:
        print(f"No images found in directory: {input_dir}")
        return
    
    for image_path in tqdm(image_files, desc=f"Processing images in {input_dir}"):
        try:
            # Initialize the PreProcess object
            processor = PreProcess(image_path)
            
            # Step 1: Crop the image
            cropped = processor.crop_image(half_width=200,left_offset=0.15,right_offset=0.6)
            
            # Step 2: Apply bilateral filtering
            bilateral = processor.apply_bilateral_filter(d=3, sigmaColor=75, sigmaSpace=75)
            
            # Step 3: Enhance contrast using CLAHE
            contrast = processor.apply_clahe(clipLimit=2, tileGridSize=(8, 8))
            
            # Step 4: Sharpen the image (using Laplacian-based sharpening)
            sharp = processor.sharpen_image(weight=1.75)
            
            # Step 5: Apply multi-scale Sobel edge detection
            edges = processor.apply_sobel_edge(scales=[3, 5, 7])
        
            # Step 6: Apply morphological closing
            skeleton = processor.apply_morph_skeletonize(kernel=(2,2))

            # Prepare the output file path for skeleton image
            base_name = os.path.basename(image_path)
            output_file = Path(settings.BASE,output_dir, 'processed_skeleton', base_name)
            cv2.imwrite(output_file, skeleton)

            # Prepare the output file path for overlay image
            output_file = Path(settings.BASE,output_dir, 'processed_overlay', base_name)
            overlay_image_sample = processor.overlay_fault_skeleton(cropped,skeleton)
            cv2.imwrite(output_file, overlay_image_sample)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    
if __name__ == "__main__":
    main()

#for cracks : sobel edge kernel sizes= (3,5,7); laplacian weight = 1.25 to increase the sharpness to extract vertical boundaries of cracks, morph closing kernel = (3,3)
# for flakings : same
# for shellings : same but morphological closing kernel = (4,4) since shelling patterns are larger than cracks and flakings
# for spallings : same but morpolohical closing kernel = (3,3) 
# for squats : laplacian weight = 1.5, morpolohical closing kernel = (2,2)