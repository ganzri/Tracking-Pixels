
"""
calculates how many training samples there are per category
takes a json as input

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
    
    category = ["necessary", "functional", "analytics", "advertising"]
    category_count = [0] * 4 
    cat_blocked = [0] * 4
    pixel = [0] * 4
    fb = 0
    for key in in_pixels:
        sample = in_pixels[key]
        cat = int(sample["label"])
        category_count[cat] += 1
        if sample["blocked"][0] == 1 or sample["blocked"][1] == 1:
            cat_blocked[cat] += 1
        if sample["img_size"][0] == 1 and sample["img_size"][1] == 1:
            pixel[cat] += 1
        if cat == 3 and "facebook" in sample["url"]:
            fb += 1

    for i in range(4):
        print(f"{category[i]}:count {category_count[i]} blocked {cat_blocked[i]} 1x1 {pixel[i]}")
    
    print(f"facebook in advertising: {fb}")

if __name__ == "__main__":
    exit(main())
