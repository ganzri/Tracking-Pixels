# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Takes a trained model and validation data and predicts labels for the validation data. Then compares these to ground truth,
creating a confusion matrix and saving a plot thereof. It also compares the ground truth and predictions to filter list decisions
(to block or not to block), and saves the correspondinng figures. Finally outputs and saves random samples of each interesting category, i.e.,
- ground truth necessary predicted analytics or advertising 
- ground truth analytics or advertising predicted necessary
- ground truth necessary blocked by at least one filterlist
- ground truth analytics and advertising not blocked by either list
- predicted necessary blocked by at least one filterlist
- predicted analytics and advertising not blocked by either list

It needs as input a trained model, the validation data as sparse matrix (after feature extraction) and as json format (before)
Using the json means that we can later easily analyse the samples (they are human readable).

Usage:
    predict_and_analyse.py  <model> <val_data_sparse> <val_data_json> 
"""
from docopt import docopt
import numpy as np
import xgboost as xgb
import pickle
import os
import json
from typing import Dict, Any
from utils import load_data
import random

#for the plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# three plot functions one for each of the figures.
def plot_gt_pred(matrix) -> None:
    #plot raw data
    df_cm = pd.DataFrame(matrix, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')])

    plt.figure(figsize=(5,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=True, fmt='', cmap='YlGnBu', cbar=True )
    plt.xlabel('Prediction')
    plt.ylabel('Ground truth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')

    plt.savefig('./GT_PRED_conf_matrix_nr.pdf', bbox_inches='tight')
    plt.clf()

    #plot percentage
    for i in range(4):
        row_tot = sum(matrix[i])
        for j in range(4):
            matrix[i][j] /= row_tot

    labels = np.array([[f'{v:.4f}'.lstrip('0')[:5] for v in row] for row in matrix])

    df_cm = pd.DataFrame(matrix, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')])

    plt.figure(figsize=(5,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
    plt.xlabel('Prediction')
    plt.ylabel('Ground truth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')

    plt.savefig('./GT_PRED_conf_matrix.pdf', bbox_inches='tight')

def plot_gt_blocked(matrix_2)-> None:
    #plot actual numbers of entries
    df_cm = pd.DataFrame(matrix_2, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])

    plt.figure(figsize=(3,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=True, fmt='', cmap='YlGnBu', cbar=True)
    plt.xlabel('Filter List')
    plt.ylabel('Ground truth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.savefig('./GT_FL_conf_matrix_nr.pdf', bbox_inches='tight')

    #plot percentage
    for i in range(4):
        row_tot = sum(matrix_2[i])
        for j in range(2):
            matrix_2[i][j] /= row_tot

    labels = np.array([[f'{v:.4f}'.lstrip('0')[:5] for v in row] for row in matrix_2])

    df_cm = pd.DataFrame(matrix_2, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])

    plt.figure(figsize=(3,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
    plt.xlabel('Filter List')
    plt.ylabel('Ground truth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.savefig('./GT_FL_conf_matrix.pdf', bbox_inches='tight')


def plot_pred_blocked(matrix_3)->None:
    #plot actual numbers
    df_cm = pd.DataFrame(matrix_3, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])

    plt.figure(figsize=(3,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=True, fmt='', cmap='YlGnBu', cbar=True)
    plt.xlabel('Filter List')
    plt.ylabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.savefig('./PRED_FL_conf_matrix_nr.pdf', bbox_inches='tight')

    #plot percentage
    for i in range(4):
        row_tot = sum(matrix_3[i])
        for j in range(2):
            matrix_3[i][j] /= row_tot

    labels = np.array([[f'{v:.4f}'.lstrip('0')[:5] for v in row] for row in matrix_3])

    df_cm = pd.DataFrame(matrix_3, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])

    plt.figure(figsize=(3,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
    plt.xlabel('Filter List')
    plt.ylabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.savefig('./PRED_FL_conf_matrix.pdf', bbox_inches='tight')

def extract_random_samples(N, keys, samples, name) -> None:
    """
    extracts N random samples and saves them in a json file with "name.json"
    :param N: nr of random samples to extract
    :param keys: contains all the keys of possible samples, from this the N samples are drawn
    :param samples: dict with the actual samples
    :param name: name of output file
    
    """
    random.seed(355187)
    if len(keys) > N:
        selected = random.sample(keys, N)
    else:
        selected = keys
    
    out: Dict[str, Dict[str, Any]] = dict()
    for i in selected:
        out[i] = samples[i]
    outname = "./to_analyse/"+ name + ".json"
    with open(outname, 'w') as dff:
        json.dump(out, dff)

def main() -> int:
    argv = None
    cargs = docopt(__doc__, argv=argv)

    #get arguments and sanity-check
    model_path: str = cargs["<model>"]
    if not os.path.exists(model_path):
        print(f"{model_path} is not a valid path.")
        return 2

    val_data_sparse: str = cargs["<val_data_sparse>"]
    if not os.path.exists(val_data_sparse):
        print(f"{val_data_sparse} is not a valid path.")
        return 2

    val_data_json: str = cargs["<val_data_json>"]
    if not os.path.exists(val_data_json):
        print(f"{val_data_json} is not a valid path")
        return 2
    
    #load model and data
    model = xgb.Booster(model_file=model_path)
    data, labels, weights = load_data(val_data_sparse)
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    with open(val_data_json) as fd:
        in_pixels = json.load(fd)
    dval: xgb.DMatrix = xgb.DMatrix(data=data, label=labels, weight=weights)

    prediction = model.predict(dval)
    
    N = len(prediction)
    if N != len(in_pixels):
        print(f"lengths don't match: prediction length {N} and json raw data: {len(in_pixels)}")
        return 2

    cat_pred = [0] * N
    
    for i in range(N):
        cat_pred[i] = np.argmax(prediction[i])
    
    json_labels = [0] * N
    is_blocked = [0] * N
    keys = [""] *N #this contains all the keys of the samples in the json dict, so we can later look them up with an index
    index = 0
    for key in in_pixels:
        sample = in_pixels[key]
        json_labels[index] = sample["label"]
        keys[index] = key
        if sample["blocked"][0] == 1 or sample["blocked"][1] == 1:
            is_blocked[index] = 1
        index += 1

    #This is done only to verify that the order of the samples is the same and that we can truly compare json and sparse sample
    #at the same index. Hopefully we never see this error msg
    for i in range(N):
        if json_labels[i] != labels[i]:
            print(f"the list are not in the same order, this approach does not work")
            return 2

    #for i in range(10):
        #print(f"Prediction: {prediction[i]}, category predicted argmax: {cat_pred[i]}, label: {labels[i]}, json: {json_labels[i]}, is blocked: {is_blocked[i]}")
    
    #confusion matrix btw ground truth and prediction
    matrix = [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]
    
    #ground truth vs is blocked
    matrix_2 = [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0]]

    #predicted vs is blocked
    matrix_3 = [[0, 0],
            [0, 0],
            [0, 0],
            [0, 0]]

    for i in range(N):
        matrix[labels[i]][cat_pred[i]] += 1
        matrix_2[labels[i]][is_blocked[i]] += 1
        matrix_3[cat_pred[i]][is_blocked[i]] += 1
    
    plot_gt_pred(matrix)
    plot_gt_blocked(matrix_2)
    plot_pred_blocked(matrix_3)
    
    #get 200 random samples from
    #Note for the thesis this was once run with 20, then 200, then sometimes all samples, as otherwise
    #we would not have had enough samples to analyse.
    random.seed(355187)
    #ground truth necessary predicted as analytics or advertising
    #ground truth analytics or advertising predicted as necessary
    nec_aa = []
    aa_nec = []
    for i in range(N):
        if labels[i] == 0 and (cat_pred[i] == 2 or cat_pred[i] == 3):
            nec_aa.append(keys[i])
        if cat_pred[i] == 0 and (labels[i] == 2 or labels[i] == 3):
            aa_nec.append(keys[i])
    extract_random_samples(200, nec_aa, in_pixels, "nec_pred_as_aa_200")
    extract_random_samples(200, aa_nec, in_pixels, "aa_pred_as_nec_200")

    #ground truth necessary blocked
    #ground truth Advertising/ Analytics not blocked
    gt_nec_bl = []
    gt_aa_nb = []
    for i in range(N):
        if labels[i] == 0 and is_blocked[i] == 1:
            gt_nec_bl.append(keys[i])
        if (labels[i] == 2 or labels[i] == 3) and is_blocked[i] == 0:
            gt_aa_nb.append(keys[i])
    extract_random_samples(200, gt_nec_bl, in_pixels, "gt_nec_blocked_200")
    extract_random_samples(200, gt_aa_nb, in_pixels, "gt_aa_not_blocked_200")

    #predicted necessary blocked
    #predicted advertising/ analytics not blocked
    pred_nec_bl = []
    pred_aa_nb = []
    pred_fc_nb = []
    for i in range(N):
        if cat_pred[i] == 0 and is_blocked[i] == 1:
            pred_nec_bl.append(keys[i])
        if (cat_pred[i] == 2 or cat_pred[i] == 3) and is_blocked[i] == 0:
            pred_aa_nb.append(keys[i]) 
        if cat_pred[i] == 1 and is_blocked[i] == 0:
            pred_fc_nb.append(keys[i])
    extract_random_samples(200, pred_nec_bl, in_pixels, "pred_nec_blocked_200")
    extract_random_samples(200, pred_aa_nb, in_pixels, "pred_aa_not_blocked_200")
    extract_random_samples(200, pred_fc_nb, in_pixels, "pred_fc_not_blocked_200")

    return 0

if __name__ == "__main__":
    exit(main())
