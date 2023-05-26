import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
import pickle

# Where are the files
file_data = 'data' + os.sep +'selected_images.csv'
df = pd.read_csv(file_data)
label = np.array(df['diagnostic'])


# Where did we store the features?
file_features = 'features/features.csv'
feature_names = [
    "Asymmetry_Score_1",
    "Asymmetry_Score_2",
    "Border_Irregularity_Score",
    "Color_Asymmetry_Score_1",
    "Color_Asymmetry_Score_2",
    "Texture_Contrast_Score",
    "Texture_Dissimilarity_Score"
]

# Load the features - remember the example features are not informative
df_features = pd.read_csv(file_features)

# Select only the desired features
selected_features = [
    "Asymmetry_Score_1",
    "Asymmetry_Score_2",
    "Border_Irregularity_Score",
    "Color_Asymmetry_Score_1",
    "Color_Asymmetry_Score_2"
]
x_selected = np.array(df_features[selected_features])

# Make the dataset, you can select different classes (see task 0)
x = np.array(df_features[feature_names])
y = label == 'MEL'  # now True means Melanoma, False means something else

# Prepare the train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Different classifiers to test out
classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(5),
    tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]),
    KNeighborsClassifier(5)  # 4th classifier using selected features
]
num_classifiers = len(classifiers)

acc_test = np.empty([num_classifiers])

for j, clf in enumerate(classifiers):
    if j == 3:  # the last classifier uses selected features
        x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)


    if isinstance(clf, tf.keras.Sequential):
        clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        clf.fit(x_train, y_train, epochs=10, verbose=0)
        y_pred = (clf.predict(x_test) > 0.5).flatten().astype(bool)
    else:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
    acc_test[j] = accuracy_score(y_test, y_pred)

print('Classifier 1 test accuracy={:.3f} '.format(acc_test[0]))
print('Classifier 2 test accuracy={:.3f} '.format(acc_test[1]))
print('Classifier 3 test accuracy={:.3f} '.format(acc_test[2]))
print('Classifier 4 (selected features) test accuracy={:.3f} '.format(acc_test[3]))  # new print statement

# Let's say you now decided to use the 5-NN with selected features
classifier = KNeighborsClassifier(5)

# It will be tested on external data, so we can try to maximize the use of our available data by training on 
# ALL of the selected features and y
x_selected = np.array(df_features[selected_features])  # re-compute x_selected in case it has been changed
classifier = classifier.fit(x_selected, y)

# This is the classifier you need to save using pickle, add this to your zip file submission
filename = 'groupXY_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))
