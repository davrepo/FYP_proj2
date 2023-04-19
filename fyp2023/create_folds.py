import os
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    input_path = "../data"
    df = pd.read_csv(os.path.join(input_path, "metadata.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    # why stratified? 
    # because we want to make sure that the distribution of the target variable is the same in each fold
    kf = model_selection.StratifiedKFold(n_splits=5) 
    for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold_
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)