"""
calculates how many training samples are per category
list, which easylist to use



Usage:
    training_validation_split.py <data_path>
"""

import json
from typing import List, Dict, Any, Tuple
from docopt import docopt
import os
import re
import random

def main() -> None:
    argv = None
    cargs = docopt(__doc__, argv=argv)
    
    dat_path: str = cargs["<data_path>"]
    random.seed(355187)

    if not os.path.exists(dat_path):
        print(f"{dat_path} is not a valid data path")
        return 
    
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    training: Dict[str, Dict[str, Any]] = dict()
    validation: Dict[str, Dict[str, Any]] = dict()

    with open(dat_path) as fd: 
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")

    category = ["necessary", "functional", "analytics", "advertising"]
    category_count = [0] * 4
    training_indices = [0]*4
    
    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        category_count[cat] += 1
    
    for i in range(4):
        tr_count =round(category_count[i]*0.8) #how many training samples integer division
        print(f"{category_count[i]} tr_count: {tr_count}")
        a = list(range(0,category_count[i]))
        tra = random.sample(a, tr_count)
        tr = sorted(tra)
        training_indices[i] = tr
    
    cat_curr = [0] * 4

    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        if len(training_indices[cat]) > 0 and cat_curr[cat] == training_indices[cat][0]: 
            #if i am sample nr x and x is in the training set
            training[key] = sample
            training_indices[cat] = training_indices[cat][1:]
        else:
            validation[key] = sample
        cat_curr[cat] += 1
    
    
    print(f"{len(training)} training samples returned")
    print(f"{len(validation)} validation samples returned")
    
    #to check this looks correct:
    val_cat = [0]*4
    for key in validation:
        cat = validation[key]["label"]
        val_cat[cat]+=1

    tr_cat = [0]*4
    for key in training:
        cat = training[key]["label"]
        tr_cat[cat] +=1


    json_outfile = "./training_data_output_offline_model/"
    with open(json_outfile+"validation.json", 'w') as fd:
        json.dump(validation, fd)
    with open(json_outfile+"training.json", 'w') as dff:
        json.dump(training, dff)

    for i in range(4):
        print(f"{category[i]}: Nr of samples: {category_count[i]}, training: {tr_cat[i]}, validation: {val_cat[i]}, tr + cat {val_cat[i] + tr_cat[i]}")

if __name__ == "__main__":
    exit(main())
