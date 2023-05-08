import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from skimage import morphology
from pandas import read_csv
import seaborn
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split


def plots(features):
    plot = seaborn.pairplot(features, hue="diagnosis", height=1, diag_kind="hist")
    plot.tight_layout()
    plt.show()


def splitDataIntoTrainTest(X, y):

    """
    Wrapper around the scikit function train_test_split.
    The goal of this function is to split properly a given dataset into training and test data.
    :X: DF containing only features
    :y: pd series containing binary values
    :return: dataframes and series splitted according to the given criteria.
    """

    # Split the given data according to given criteria
    # * random state --> for reproducibility
    # * stratify --> makes sure that the distribution of cancer is in each equal
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)

    # Return the result
    return X_train, X_test, y_train, y_test


def featureScores(X_train, y_train, k):
    """
    This fucntion returns the selector object and importance scores and for each feature
    for a univariate, filter-based feature selection.
    :X_train: Set of input variables for testing
    :y_train: Set of output variables for testing
    :k: Number of features to be selected (integer)
    """

    selector = SelectKBest(mutual_info_classif,
                           k=k)  # Selecting top k features using mutual information for feature scoring
    selector.fit(X_train, y_train)  # fit selector to data

    # Retrieve scores
    scores = selector.scores_

    return scores, selector

def main():
    norm = read_csv("features_normalized.csv")
    feat = read_csv("features_output.csv")
    numeral = read_csv("features_numeral.csv")
    columns_with_features = list(norm.columns[1:])

    X_train, X_test, y_train, y_test = splitDataIntoTrainTest(norm[columns_with_features],
                                                                             norm.iloc[:, 0])

    X_train_with_Y = X_train.copy()
    X_train_with_Y["diagnosis"] = y_train
    seaborn.pairplot(X_train_with_Y, hue="diagnosis", height=3, diag_kind="hist")
    plt.show()

    #print(norm.iloc[:, 0])
    feature_scores, selector = featureScores(norm[columns_with_features], norm.iloc[:, 0], k=2)
    features_mel = len(feature_scores)

    # Visualize feature scores
    plt.bar(np.arange(0, features_mel), feature_scores, width=.2)
    plt.xticks(np.arange(0, features_mel), list(X_train.columns), rotation='vertical')
    plt.show()

    # Select the two best features based on the selector
    X_train_adj = selector.transform(X_train)
    X_test_adj = selector.transform(X_test)
    print(X_test_adj)
    print(X_train_adj)

if __name__ == "__main__":
    main()