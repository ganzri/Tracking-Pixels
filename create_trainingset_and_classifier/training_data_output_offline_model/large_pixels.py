# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
To answer (partially) the question whether there are large pixels (larger than 1x1).

counts how many samples are matched to a consent declaration and larger than 1x1, specifically also counts the pinkgellac ones as these stem from missmatching (see thesis, 5.3.1), counts some common 2x2 pixels.
"""

import json
from typing import Dict, Any, List
from urllib import parse
#from docopt import docopt

def main() -> None:
    #argv = None
    #cargs = docopt(__doc__, argv=argv)

    #mode: str = cargs["<mode>"]
    #to_query: str = cargs["<query>"]

    in_pixels: Dict[str, Dict[str, Any]] = dict()

    with open('10_12_2021_filter_lists.json') as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")

    count1 = 0
    count2 = 0
    count3 = 0
    pink_count = 0
    adobe = 0
    o7 = 0 

    for key in in_pixels:
        sample = in_pixels[key]
        if sample["matched"] == 1 and (sample["img_size"][0] > 1 or sample["img_size"][1] > 1):
            #print(sample)
            count1 += 1
            if "pinkgellac.nl" in sample["url"]:
                pink_count += 1
        elif sample["matched"] == 0 and (sample["blocked"][0] == 1 or sample["blocked"][1] == 1) and (sample["img_size"][0] == 2 and sample["img_size"][1] == 2):
            url = sample["url"]
            if "omtrdc.net" in url:
                adobe += 1
            elif "2o7.net" in url:
                o7 += 1
            #else:
                #print(sample)
            count2 += 1
        elif sample["matched"] == 0 and (sample["img_size"][0] == 1 and sample["img_size"][1] == 1) and (sample["blocked"][0] == 0 and sample["blocked"][1] == 0):
            print(sample)
            count3 += 1

    print(f"matched large images: {count1}")
    print(f"pinkgellac.nl count: {pink_count}")
    print(f"unmatched, blocked 2x2 images: {count2}")
    print(f"adobe: {adobe}, 07: {o7}")
    print(F"unmatched 1x1 not blocked: {count3}")

if __name__ == "__main__":
    exit(main())

