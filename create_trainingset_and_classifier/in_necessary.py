# Copyright (C) 2022, Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
calculates how many training samples from this url are found in the necessary category, and prints that.
This script was used to analyse samples manually, and specifically to test whether samples with ground truth
analytics or advertising and prediction necessary have a very high number of samples that could not be
matched and were therefore put into the necessary category. This would explain why they are misclassified.

<data_path> is the path to the json with all the training samples
<url_query> is the domain, netloc, netloc plus path (any part of an url), we want to know whether it is found often in necessary.

Usage:
    analyse.py <data_path> <url_query>
"""

import json
from typing import List, Dict, Any, Tuple
from docopt import docopt
import os


def main() -> None:
    argv = None
    cargs = docopt(__doc__, argv=argv)

    dat_path: str = cargs["<data_path>"]
    query: str = cargs["<url_query>"]

    if not os.path.exists(dat_path):
        print(f"{dat_path} is not a valid data path")
        return

    in_pixels: Dict[str, Dict[str, Any]] = dict()
    with open(dat_path) as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")
    
    count = 0
    nec_unm = 0
    for k in in_pixels:
        sample = in_pixels[k]
        url = sample["url"]
        if query in url:
            count += 1
            if sample["matched"] == 0 and sample["label"] == 0:
                nec_unm += 1

    print(f"the query {query} occurs in a total of {count} samples, {nec_unm} of which are from necessary unmatched")

if __name__ == "__main__":
    exit(main())
                    
