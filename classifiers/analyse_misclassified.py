from scipy.sparse import csr_matrix
import pickle
import os

from typing import Union, Optional, List, Callable

from statistics import mean, stdev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    features, labels = load_data("./xgb_predict_stats/validation_matrix_20220113_132202.sparse")
    preds_df = pd.read_csv("./xgb_predict_stats/softprob_predictions_20220113_132202.csv")

    #sanity check: its the same order in both cases .csv and .sparse and .sparse.labels
    N = len(labels)
    for i in range(N):
        #print(f"sparse label: {labels[i]}, prediction label: {preds_df.labels[i]}")
        if labels[i] != preds_df.labels[i]:
            print(f"not equal: {i}")
    
    miscl_ind = []
    t_l = []
    max_l = []
    feat_ind = []
    feat_data = []
    c_0as = [0]*4
    dom_2 = [0] * 500
    dom_3 = [0] * 500
    c_1x1 = 0
    url_entropy = []
    url_entropy_2 = []
    url_entropy_3 = []
    tp_0 = 0
    tp_mc = 0
    
    """
    #check for non 1x1 in tracking pixels:
    for index, row in preds_df.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        idx = features[index].indices
        if 2002 not in idx and not (true_label == 0 and maxprob_label == 0):
            print(f"{index}, is not 1x1, true label: {true_label}, classified as: {maxprob_label}")
            print(idx)
            print(features[index].data)
    """
    for index, row in preds_df.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        idx = features[index].indices
        values = features[index].data
        if true_label != maxprob_label:
            print(idx)
            print(values)


    """
    for index, row in preds_df.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        #print(f"{index}, true_label: {true_label}, maxprob_label: {maxprob_label}")
        if true_label == maxprob_label and true_label == 0:
            url_entropy.append(features[index].data[-2])
            idx = features[index].indices
            if (2000 in idx) and (2002 in idx):
                tp_0 += 1
        elif true_label != maxprob_label:
            #print(f"{index}, true_label: {true_label}, maxprob_label: {maxprob_label}")
            miscl_ind.append(index)
            t_l.append(true_label)
            max_l.append(maxprob_label)
            feat_ind.append(features[index].indices)
            feat_data.append(features[index].data)
            if true_label == 0:
                c_0as[maxprob_label] += 1
                idx = features[index].indices
                data = features[index].data
                if (2000 in idx) and (2002 in idx):
                    tp_mc +=1
                if 2002 in idx:
                    c_1x1 += 1
                    if maxprob_label == 1:
                        print("functional is 1x1")
                if maxprob_label == 2:
                    if idx[0] < 500:
                        dom_2[idx[0]] += 1
                    se_url = data[-2]
                    url_entropy_2.append(se_url)
                    #if se_url > 5.0 and idx[0] != 25:
                        #print(idx)
                        #print(data[-6:])
                if maxprob_label == 3:
                    #print(f"{idx[:-6]}")
                    if idx[0] < 500:
                        dom_3[idx[0]] += 1
                        se_url = data[-2]
                        url_entropy_3.append(se_url)
    """        
    """
                #filter 1 + 2
                else:
                    print(idx)
                if idx[0] < 500:
                    if maxprob_label == 3:
                        dom_3[idx[0]] += 1
                        if idx[0] == 2: #for google.com
                            print(idx)
                    elif maxprob_label == 2:
                        dom_2[idx[0]] += 1
    """
    
    """
    print(c_0as)
    print(f"in total {sum(c_0as)} samples are misclassified")
    print(dom_2)
    for i in range(500):
        if dom_2[i] != 0:
            print(f"{i}: {dom_2[i]}")
    print(dom_3)
    for i in range(500):
        if dom_3[i] != 0:
            print(f"{i}: {dom_3[i]}")
    print(f"nr of 1x1 pixels in misclassified: {c_1x1}")
    print(f"nr of third party elements in 'truely' necessary: {tp_0}, in misclassified: {tp_mc}")
    """
    print(len(url_entropy))
    plt.hist(url_entropy, 100)
    plt.title("Shannon Entropy in gstatic")
    plt.savefig("entropy_url.png")
    plt.clf()
    """
    plt.hist(url_entropy_2, 100)
    plt.title("Shannon Entropy in URL classified as Analytics")
    plt.savefig("entropy_url_2.png")
    plt.clf()
    plt.hist(url_entropy_3, 100)
    plt.title("Shannon Entropy in URL classified as Advertising")
    plt.savefig("entropy_url_3.png")
    plt.clf()

    """
    """
    #print(miscl_ind)
    miscl = pd.DataFrame(miscl_ind, columns=["index"])
    miscl.insert(1, "true_label", t_l)
    miscl.insert(2, "maxprob_label", max_l)
    miscl.insert(3, "feature_indices", feat_ind)
    miscl.insert(4, "feature_data", feat_data)
    #print(miscl)
    """
    print("reached the end")
    return 0


if __name__ == "__main__":
    exit(main())
