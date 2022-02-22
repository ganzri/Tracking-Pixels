# Copyright (C) 2022, Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
prints the domain number as in the resource ranking of samples in necessary which are most commonly misclassified from the 
500 top domains, plus how often they are misclassified. 
This was used to decide which domains to move (see thesis appendix for the decision), these were then reclassified in 
training_data_output_offline/reclassify.py
"""

from scipy.sparse import csr_matrix
import pickle
import os

from typing import Union, Optional, List, Callable

from statistics import mean, stdev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(dpath: str):
    """ 
    data loading function based on Dino's in utils
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
    features, labels = load_data("./xgb_predict_stats/validation_matrix_20220202_143638.sparse") #validation matrix from one of the 
    #five folds of CV, from an early iteration of the model before the data was filtered.
    preds_df = pd.read_csv("./xgb_predict_stats/softprob_predictions_20220202_143638.csv")
    #predictions for this validation matrix

    #sanity check: its the same order in both cases .csv and .sparse and .sparse.labels
    N = len(labels)
    for i in range(N):
        #print(f"sparse label: {labels[i]}, prediction label: {preds_df.labels[i]}")
        if labels[i] != preds_df.labels[i]:
            print(f"not equal: {i}")
    
    c_0as = [0]*4
    dom_2 = [0] * 500
    dom_3 = [0] * 500
    
    for index, row in preds_df.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        idx = features[index].indices
        if true_label != maxprob_label and true_label == 0:
            c_0as[maxprob_label] += 1
            idx = features[index].indices
            data = features[index].data
            if maxprob_label == 2:
                if idx[0] < 500:
                    dom_2[idx[0]] += 1
            if maxprob_label == 3:
                if idx[0] < 500:
                    dom_3[idx[0]] += 1


    print(c_0as)
    print(f"in total {sum(c_0as)} samples are misclassified")
    print(f"necessary classified as 2:")
    for i in range(500):
        if dom_2[i] != 0:
            print(f"{i}: {dom_2[i]}")
    print("necessary classified as 3:")
    for i in range(500):
        if dom_3[i] != 0:
            print(f"{i}: {dom_3[i]}")
    
    return 0


if __name__ == "__main__":
    exit(main())
