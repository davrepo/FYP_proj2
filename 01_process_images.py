"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import packages for image processing
from skimage import morphology #for measuring things in the masks

#-------------------
# Help functions
#-------------------


#Main function to extract features from an image, that calls other functions    
def extract_features(image):
    
    [r,g,b] = get_pixel_rgb(image)

    #Here you need to add more of your custom-made functions for measuring features!

    return np.array([r,g,b], dtype=np.float16)




#Extracts the RGB values at location 100, 100. Example feature, probably not very useful
def get_pixel_rgb(image):

    x_coord = 100
    y_coord = 100

    r = image[x_coord, y_coord, 0]
    g = image[x_coord, y_coord, 1]
    b = image[x_coord, y_coord, 2]

    return r,g,b



#-------------------
# Main script
#-------------------


#Where is the raw data
file_data = '..' + os.sep + 'data' + os.sep +'metadata.csv'
path_image = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'imgs_part_1'


#Where we will store the features
file_features = 'features/features.csv'


#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data
image_id = list(df['img_id'])
label = np.array(df['diagnostic'])

# Here you could decide to filter the data in some way
is_nevus =  label == 'NEV'


#For now I just take the first 100...
num_images = 100


#Make array to store features
feature_names = ['red','green','blue']
num_features = len(feature_names)
features = np.empty([num_images,num_features], dtype=np.float16)  

#Loop through all images
for i in np.arange(num_images):
    
    # Define filenames related to this image
    file_image = path_image + os.sep + image_id[i] 
    
    if exists(file_image):
        # Read the image
        im = plt.imread(file_image)
        im = np.float16(im)  
    
        # Measure features - this does not do anything too useful yet!
        x = extract_features(im) 
           
        # Store in the variable we created before
        features[i,0:num_features] = x



#Save the features to a file      
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  
    