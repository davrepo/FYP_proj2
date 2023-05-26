# FYP_proj2
ITU First Year Project (Project 2)

```
├── data
│   ├── images
|         └── imgs_part_1  <- selected 138 images only
|   └── shanon_masks_total  <- Masks in png format      
│   └── metadata.csv   
|   └── selected_images.csv <- Selected images for training and testing
│
├── fyp2023            <- Project repository
│   └── 00_total.ipynb  <- Main notebook, contains all the code for the project to run from start to finish
|   └── extract_features.py   <- helper functions
|   └── 01_process_images.py
|   └── 02_train_classifiers.py
|   └── 03_evaluate.classifier.py    <- TA should run this file to evaluate the classifier
```

NB! In 03_evaluate.classifier.py, the classify(img, mask) function accepts only image objects as parameters and not input from plt.imread.

classify_path(img_path, mask_path) accepts image paths as parameters.

## How to run the project
```
from PIL import Image

image_path = '..' + os.sep + 'FYP_proj2' + os.sep + 'data' + os.sep +'images_resized' + os.sep + 'PAT_710_1330_243.PNG'
mask_path = '..' + os.sep + 'FYP_proj2' + os.sep + 'data' + os.sep + 'shanon_masks_total' + os.sep + 'PAT_710_1330_243.PNG'

img = Image.open(image_path)
mask = Image.open(mask_path)

classify(img, mask)
classify_path(image_path, mask_path)
```
