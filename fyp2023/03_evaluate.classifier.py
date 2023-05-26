import os
import cv2
import pickle #for loading your trained classifier
from PIL import Image

from extract_features import extract_features #our feature extraction


# image_path = '..' + os.sep + 'FYP_proj2' + os.sep + 'data' + os.sep +'images_resized' + os.sep + 'PAT_710_1330_243.PNG'
# mask_path = '..' + os.sep + 'FYP_proj2' + os.sep + 'data' + os.sep + 'shanon_masks_total' + os.sep + 'PAT_710_1330_243.PNG'

# img = Image.open(image_path)
# mask = Image.open(mask_path)


# The function that should classify new images. 
# The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask):
     
     selected_features_indices = [0, 1, 2, 3, 4]
   
     #Extract features (the same ones that you used for training)
     x = extract_features(img, mask)
     
     # Select only the desired features
     x_selected = x[selected_features_indices].reshape(1, -1)
     
     #Load the trained classifier
     classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))
    
     #Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(x_selected)
     pred_prob = classifier.predict_proba(x_selected)
     
     print('predicted label is ', pred_label)
     print('predicted probability is ', pred_prob)
     return pred_label, pred_prob

def classify_path(img_path, mask_path):
          
     # Load the image and mask from the path
     img = Image.open(image_path)
     mask = Image.open(mask_path)
    
     selected_features_indices = [0, 1, 2, 3, 4]
   
     #Extract features (the same ones that you used for training)
     x = extract_features(img, mask)
     
     # Select only the desired features
     x_selected = x[selected_features_indices].reshape(1, -1)
         
     
     #Load the trained classifier
     classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))
    
     #Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(x_selected)
     pred_prob = classifier.predict_proba(x_selected)
     
     
     print('predicted label is ', pred_label)
     print('predicted probability is ', pred_prob)
     return pred_label, pred_prob


# classify(img, mask)
# classify_path(image_path, mask_path)
    
# The TAs will call the function above in a loop, for external test images/masks