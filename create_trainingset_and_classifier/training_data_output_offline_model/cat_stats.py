# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
calculates how many training samples are per category

Usage:
    cat_stats.py <data_path>
"""

import json
from typing import List, Dict, Any, Tuple
from collections import Counter
from docopt import docopt
import os


def main() -> None:
    argv = None
    cargs = docopt(__doc__, argv=argv)
    
    dat_path: str = cargs["<data_path>"]
    if not os.path.exists(dat_path):
        print(f"{dat_path} is not a valid data path")
        return 
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    
    with open(dat_path) as fd: 
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")
    
    category = ["necessary", "functional", "analytics", "advertising", "necessary declared"]
    category_count = [0] * 5 
    cat_blocked = [0] * 5
    big_blocked = 0
    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        if cat == 0 and sample["matched"] == 1: #count how many necessary come from being matched, vs added
            category_count[4] += 1
            if sample["blocked"][0] == 1:
                cat_blocked[4] += 1
            
        category_count[cat] += 1
        if sample["blocked"][0]:
            cat_blocked[cat] += 1
            if cat == 0 and (sample["img_size"][0] > 1 or sample["img_size"][1] > 1):
                big_blocked += 1 
    
    print(category_count[:4])
    total = sum(category_count[:4])

    for i in range(5):
        print(f"{category[i]}: sample count {category_count[i]} percentage of total: {category_count[i]/total} how many are blocked {cat_blocked[i]}")
    print(f"nr of big images blocked: {big_blocked}")

if __name__ == "__main__":
    exit(main())
