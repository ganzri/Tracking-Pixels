
"""
Script to display one tree at a time after the model was run.
For four classes tree nr 0, 4, 8 etc are trees for class 0 (necessary)
1, 5, 9 etc are trees for class 1 (functional) etc
Note: change i to q in feature map text file for quantitative features to see the values used for the decision in the tree

Usage:
    trees_display.py <model> <feature_map> <tree_nr>
"""
import matplotlib.pyplot as plt
from sklearn import tree
from docopt import docopt
import numpy as np
import xgboost as xgb
import pickle
import os

def main() -> int:
    argv = None
    cargs = docopt(__doc__, argv=argv)

    model_path: str = cargs["<model>"]
    if not os.path.exists(model_path):
        print(f"{model_path} is not a valid path.")
        return 2

    feature_path: str = cargs["<feature_map>"]
    if not os.path.exists(feature_path):
        print(f"{feature_path} is not a valid path.")
        return 2
    class_names = ["necessary", "functional", "analytics", "advertising"]

    tid: int=int(cargs["<tree_nr>"])

    model = xgb.Booster(model_file=model_path)
    with open(feature_path, 'r') as fd:
        features = fd.readlines()

    tree_c = 0
    for i in model:
        tree_c += 1
    print(tree_c)

    xgb.plot_tree(model, fmap = feature_path, num_trees = tid, class_name = class_names)
    plt.show()
    return 0

if __name__ == "__main__":
    exit(main())
