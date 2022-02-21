"""
calculates how many training samples are per category
list, which easylist to use



Usage:
    analyse.py <data_path>
"""

import json
from typing import List, Dict, Any, Tuple
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
    
    i = 1
    for k in in_pixels:
        print(i)
        i += 1
        print(in_pixels[k])
        print("\n")
    
    

if __name__ == "__main__":
    exit(main())
                    
