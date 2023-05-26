"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# Import our own file that has the feature extraction functions
from extract_features import extract_features



#-------------------
# Main script
#-------------------


#Where is the raw data
file_data = 'data' + os.sep +'selected_images.csv'
path_image = 'data' + os.sep + 'images' + os.sep + 'imgs_part_1'
path_mask = 'data' + os.sep + 'shanon_masks_total'


#Where we will store the features
if not os.path.exists('features'):
    os.makedirs('features')
    
file_features = 'features/features.csv'

#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data. 
image_id = list(df['img_id'])
label = np.array(df['diagnostic'])

# Here you could decide to filter the data in some way (see task 0)
# For example you can have a file selected_images.csv which stores the IDs of the files you need
num_images = len(image_id)

#Make array to store features
feature_names = ["Asymmetry_Score_1", 
                 "Asymmetry_Score_2", 
                 "Border_Irregularity_Score", 
                 "Color_Asymmetry_Score_1", 
                 "Color_Asymmetry_Score_2", 
                 "Texture_Contrast_Score", 
                 "Texture_Dissimilarity_Score"]

num_features = len(feature_names)
features = np.zeros([num_images, num_features], dtype=np.float16)


#Loop through all images
for i in np.arange(num_images):
    
    # Define filenames related to this image
    image_path = path_image + os.sep + image_id[i] 
    mask_path = path_mask + os.sep + image_id[i]
    
    if exists(image_path) and exists(mask_path):
        
        # Read the image
        image = Image.open(image_path)
        mask = Image.open(mask_path)
    
        # Measure features - this does not do anything useful yet!
        x = extract_features(image, mask) 
           
        # Store in the variable we created before
        features[i,:] = x
       
        
#Save the image_id used + features to a file   
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  