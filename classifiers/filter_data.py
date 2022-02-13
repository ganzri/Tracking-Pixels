from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import pickle
import os
import xgboost as xgb
from typing import Union, Optional, List, Callable

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from statistics import mean, stdev

import numpy as np
import pandas as pd


def load_data(dpath: str):
    """ data loading function based on Dino's in utils
    Supports loading from pickled sparse matrix.
    :param dpath: Path for the data to be loaded.
    :return a tuple of (features, labels), where 'features' is a sparse matrix and the others are lists.
    """
    features: Union[xgb.DMatrix, csr_matrix]
    labels: Optional[List[float]] = None
    
    with open(dpath, 'rb') as fd:
        features = pickle.load(fd)

    labels_fn = dpath + ".labels"
    if os.path.exists(labels_fn):
        with open(labels_fn, 'rb') as fd:
            labels = pickle.load(fd)

    return features, labels


def main() -> int:
    path = "../processed_features/processed_20220113_112440.sparse"
    features, labels = load_data(path)
    c_changes = 0
    for i in range(len(labels)):
        indices = features[i].indices
        old_label = labels[i]
        idx = indices[0]
        if old_label == 0 or old_label == 1: #0 to filter necessary, 1 to filter functional
            if (idx in [0, 25, 38, 101, 110, 179]) or (idx in [18, 27, 29, 67, 76, 40, 380]): #first set: very, very likely in cat 2, second could be cat 2 or 3
                labels[i] = 2
                c_changes += 1
            elif idx in [1, 5, 9, 10, 15, 34, 41, 45, 51, 58, 71, 74, 79, 88, 99, 111, 118, 119, 144, 152, 183, 197, 229, 251, 277, 291, 379, 389, 409, 446]: #fb or track.hubspot.com or doubleclick.net (different variants) or pinterest
                labels[i] = 3
                c_changes += 1
            elif idx == 2 and (((1010 in indices) and (1013 in indices)) or ((1019 in indices) and (1020 in indices))):
                #pagehead/1p-conversion and ad/ga-audiences from google.com moved to cat 3
                labels[i] = 3
                c_changes += 1
    print(f"nr of labels changed: {c_changes}")
    
    out_path = path
    with open(out_path + ".labels", 'wb') as fd:
        pickle.dump(labels, fd)

    print("reached the end")
    return 0


if __name__ == "__main__":
    exit(main())
