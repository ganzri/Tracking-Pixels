import json
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy
import matplotlib.pyplot as plt
from math import log

def main() -> None:
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    out_pixels: Dict[str, Dict[str, Any]] = dict()

    with open('first_filter.json') as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")

    #get all functional pixels
    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        if cat == 1:
            out_pixels[key] = sample
            g ="google.com"
            f ="facebook.com"
            if g not in sample["url"] and f not in sample["url"]: 
                print(sample)
    print(f"Nr of functional pixels: {len(out_pixels)}")

if __name__ == "__main__":
    exit(main())
