# FYP_proj2
ITU First Year Project (Project 2)

```
├── data
│   ├── images
|         └── imgs_part_1  <- selected 138 images
|   └── shanon_masks_total  <- Masks in png format      
│   └── metadata.csv   
|   └── selected_images.csv <- Selected images' metadata for training and testing
|   └── manual_measurement.csv <- Manual ratings of images
│
├── fyp2023            <- Project repository
│   └── 00_total.ipynb  <- Main notebook, contains all the code for the project to run from start to finish
|   └── extract_features.py   <- helper functions
|   └── 01_process_images.py
|   └── 02_train_classifiers.py
|   └── 03_evaluate.classifier.py    <- TA should run this file to evaluate the classifier
|   └── groupXY_classifier.sav  <- Saved classifier
├── groupXY_classifier.sav  <- Saved classifier
```

NB! In 03_evaluate.classifier.py, the classify(img, mask) function accepts only image objects as parameters and not input from plt.imread.

classify_path(img_path, mask_path) accepts image paths as parameters.

## How to run the project
```python
from PIL import Image

image_path = '..' + os.sep + 'FYP_proj2' + os.sep + 'data' + os.sep +'images_resized' + os.sep + 'PAT_710_1330_243.PNG'
mask_path = '..' + os.sep + 'FYP_proj2' + os.sep + 'data' + os.sep + 'shanon_masks_total' + os.sep + 'PAT_710_1330_243.PNG'

img = Image.open(image_path)
mask = Image.open(mask_path)

classify(img, mask)
classify_path(image_path, mask_path)
```

## Tables and Figures in the Report
Functions used to generate tables and figures in the report are in the notebooks below:
- a0_prelude.ipynb
- a1_image_size.ipynb
- a2_segmentation.ipynb
- a3_feature_extract.ipynb
- a4_feature_scaling_norm.ipynb
- a4a_feature_score.ipynb
- a5_model.ipynb
- a6_PCA.ipynb

## Saved classifier
2 copies of the saved classifier are located at:
- fyp2023/groupXY_classifier.sav
- groupXY_classifier.sav (the main directory)

## Function that tkaes an image/mask and outputs its probability of being suspicious
- classify(img, mask)
- classify_path(image_path, mask_path)  
located at /fyp2023/03_evaluate.classifier.py