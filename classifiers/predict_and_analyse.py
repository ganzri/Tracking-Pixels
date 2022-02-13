
"""
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
#from scipy.sparse import csr_matrix

#for the plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

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

    labels = np.array([[f'{v:.2f}'.lstrip('0')[:3] for v in row] for row in matrix])

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

    labels = np.array([[f'{v:.2f}'.lstrip('0')[:3] for v in row] for row in matrix_2])

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

    labels = np.array([[f'{v:.2f}'.lstrip('0')[:3] for v in row] for row in matrix_3])

    df_cm = pd.DataFrame(matrix_3, index=[i for i in ('Necessary', 'Functional', 'Analytical', 'Advertisement')],
                     columns=[i for i in ('not blocked', 'blocked')])

    plt.figure(figsize=(3,3))
    sn.heatmap(data=df_cm, annot_kws={'fontsize': 12}, annot=labels, fmt='', cmap='YlGnBu', cbar=True, vmin=0, vmax=1)
    plt.xlabel('Filter List')
    plt.ylabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.savefig('./PRED_FL_conf_matrix.pdf', bbox_inches='tight')


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
    url = [""] * N
    index = 0
    for key in in_pixels:
        sample = in_pixels[key]
        json_labels[index] = sample["label"]
        url[index] = sample["url"]
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
        if cat_pred[i] == 3 and is_blocked[i] == 0:
            print(url[i])

    print(matrix)
    print(matrix_2)
    print(matrix_3)
    
    plot_gt_pred(matrix)
    plot_gt_blocked(matrix_2)
    plot_pred_blocked(matrix_3)

    return 0

if __name__ == "__main__":
    exit(main())
