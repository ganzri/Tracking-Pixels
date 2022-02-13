"""
Get misclassified samples from the validation split and check why they are misclassified by loading them from the original json

validation matrix is produced by the xgb_train.py, so is softprob_pred, data refers to the original json before the feature extraction

Usage:
    misclassified.py <validation_matrix> <softprob_pred> <data>
"""
from scipy.sparse import csr_matrix
import pickle
import os
import json
from typing import Union, Optional, List, Callable

import pandas as pd
import numpy as np
from docopt import docopt

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
    argv = None
    cargs = docopt(__doc__, argv=argv)

    v_matrix: str = cargs["<validation_matrix>"]
    prob_pred: str = cargs["<softprob_pred>"]
    dat_path: str = cargs["<data>"]
    if not os.path.exists(dat_path) or not os.path.exists(prob_pred) or not os.path.exists(v_matrix):
        print(f"not a valid data path")
        return

    features, labels = load_data(v_matrix)
    preds_df = pd.read_csv(prob_pred)
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    out_pixels: Dict[str, Dict[str, Any]] = dict()
    ids_to_check: Dict[str, int] = dict()

    with open(dat_path) as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")


    #sanity check: its the same order in both cases .csv and .sparse and .sparse.labels
    N = len(labels)
    for i in range(N):
        #print(f"sparse label: {labels[i]}, prediction label: {preds_df.labels[i]}")
        if labels[i] != preds_df.labels[i]:
            print(f"not equal: {i}")
    
    
    for index, row in preds_df.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        if true_label != maxprob_label and true_label == 0:
            sample_id = features[index].data[-1]
            ids_to_check[sample_id] = maxprob_label

    print(len(ids_to_check))

    for key in in_pixels:
        sample = in_pixels[key]
        if sample["id"] in ids_to_check:
            sample["label"] = str(ids_to_check[sample["id"]])
            out_pixels[key] = sample
    
    print(f"nr of misclassified samples in necessary category in this validation set: {len(out_pixels)}")

    with open("misclassified.json", "w") as outf:
        json.dump(out_pixels, outf)


    return 0


if __name__ == "__main__":
    exit(main())
