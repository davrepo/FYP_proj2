#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:53:20 2023

@author: vech
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import packages for image processing
from PIL import Image   
from skimage import morphology  # For morphological operations
from skimage.measure import find_contours, label    # For finding contours
from skimage.morphology import binary_closing   # For closing operation
from skimage.color import rgb2lab, rgb2gray   # For color space conversion
from skimage.transform import rotate, resize    # For rotating and resizing images
from skimage.measure import regionprops   # For region properties
from skimage.feature import graycomatrix, graycoprops   # For GLCM
from skimage.util import img_as_ubyte   # For scaling images to 8-bit
from scipy.ndimage import center_of_mass, rotate    # For calculating center of mass and rotating images
from skimage.util import img_as_ubyte  # For scaling images to 8-bit



#-------------------
# Help functions
#------------------



def extract_features(image_obj, mask_obj):
    
    image = image_obj.resize((512, 512), resample=Image.BILINEAR)
    mask = mask_obj.resize((512, 512), resample=Image.BILINEAR)
    
    image = np.array(image)
    mask = np.array(mask).astype(np.uint8)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    if masked_image.shape[2] == 4:  # If image has 4 channels
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGBA2RGB)
    
    
    inverted_mask = cv2.bitwise_not(np.array(mask))
    masked_image_arr = np.array(masked_image)
    gray_image = cv2.cvtColor(masked_image_arr, cv2.COLOR_RGB2GRAY)
    inverted_mask = Image.fromarray(inverted_mask)  
    
    asymmetry_1 = asymmetry_score_1(inverted_mask)
    asymmetry_2 = asymmetry_score_2(inverted_mask)
    
    border_irregularity = border_irregularity_score(inverted_mask)

    color_asymmetry_1_score = color_asymmetry_1(masked_image)
    color_asymmetry_2_score = color_asymmetry_2(masked_image)

    texture_contrast_score = texture_score_contrast(gray_image)
    texture_dissimilarity_score = texture_score_dissimilarity(gray_image)
    
    feature_vector = np.array([asymmetry_1, asymmetry_2, border_irregularity, 
                               color_asymmetry_1_score, color_asymmetry_2_score,
                               texture_contrast_score, texture_dissimilarity_score], dtype=np.float16)

    return feature_vector


# Feature engineering functions

def asymmetry_score_1(image):
    image = image.convert('1')
    width, height = image.size

    area_top, area_bottom, area_left, area_right = 0, 0, 0, 0
    for y in range(height):
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            if pixel_value == 0:  
                if y < height // 2:
                    area_top += 1
                else:
                    area_bottom += 1

                if x < width // 2:
                    area_left += 1
                else:
                    area_right += 1

    vertical_asymmetry = abs(area_top - area_bottom) / (area_top + area_bottom)
    horizontal_asymmetry = abs(area_left - area_right) / (area_left + area_right)

    average_asymmetry = (vertical_asymmetry + horizontal_asymmetry) / 2
    return average_asymmetry


def asymmetry_score_2(image, num_rotations=10):
    image = image.convert('1')
    
    image_array = np.array(image)

    center_of_mass_y, center_of_mass_x = center_of_mass(image_array)

    min_asymmetry_score = float('inf')

    for angle in np.linspace(0, 180, num_rotations, endpoint=False):
        rotated_image = rotate(image_array, angle, reshape=False)

        area_top, area_bottom, area_left, area_right = 0, 0, 0, 0
        for y in range(rotated_image.shape[0]):
            for x in range(rotated_image.shape[1]):
                pixel_value = rotated_image[y, x]
                if pixel_value == 0:  
                    if y < center_of_mass_y:
                        area_top += 1
                    else:
                        area_bottom += 1

                    if x < center_of_mass_x:
                        area_left += 1
                    else:
                        area_right += 1

        vertical_asymmetry = abs(area_top - area_bottom) / (area_top + area_bottom)
        horizontal_asymmetry = abs(area_left - area_right) / (area_left + area_right)

        average_asymmetry = (vertical_asymmetry + horizontal_asymmetry) / 2

        min_asymmetry_score = min(min_asymmetry_score, average_asymmetry)

    return min_asymmetry_score


def border_irregularity_score(image):
    image = image.convert('1')

    mask = np.array(image)

    mask_closed = binary_closing(mask)

    contours = find_contours(mask_closed, 0.5)
    border = max(contours, key=len)

    perimeter = 0
    for i in range(len(border) - 1):
        perimeter += np.linalg.norm(border[i + 1] - border[i])

    area = np.sum(mask)

    irregularity_score = perimeter / area
    return irregularity_score


def color_asymmetry_1(image):
    
    lab_image = rgb2lab(image)

    height, width, _ = image.shape

    block_size = 3
    num_blocks_x = width // block_size
    num_blocks_y = height // block_size

    color_avgs = []

    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            block = lab_image[j*block_size:(j+1)*block_size, i*block_size:(i+1)*block_size, :]

            if np.sum(block[:, :, 0] < 90) >= (0.75 * block_size**2):
                color_avg = np.mean(block, axis=(0, 1))
                color_avgs.append(color_avg)

    variances = np.var(color_avgs, axis=0)
    variance_score = np.mean(variances)
    
    return variance_score


def color_asymmetry_2(image):

    lab_image = rgb2lab(image)

    lab_image = resize(lab_image, (256, 256), mode='reflect', anti_aliasing=True)

    center_line = lab_image.shape[1] // 2

    left_half = lab_image[:, :center_line]
    right_half = lab_image[:, center_line:]

    right_half_flipped = np.fliplr(right_half)

    diff = np.abs(left_half - right_half_flipped)

    avg_color_diff = np.mean(diff)

    return avg_color_diff



def texture_score_contrast(image):
    if image.ndim == 3:
        image = rgb2gray(image)

    image = img_as_ubyte(image)

    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]

    return contrast


def texture_score_dissimilarity(image):
    if image.ndim == 3:
        image = rgb2gray(image)

    image = img_as_ubyte(image)

    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]

    return dissimilarity