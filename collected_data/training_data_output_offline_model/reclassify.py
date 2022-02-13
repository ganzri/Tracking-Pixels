#todo remove uncomment at the end
"""
Moves pixels from 9 very common netlocs (e.g. Facebook, Google Analytics) from necessary and functional to the correct class
These 10 were choosen based on having a high ranking (low rank nr) in the top domains list and they are commonly found 
misclassified (or not classified) in necessary or functional
Usage:
    reclassify.py <data_path>
"""

import json
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy
import re
from urllib import parse
from math import log
from docopt import docopt
import os

def load_lookup_from_csv(csv_source: str, count: int) -> Dict[str, int]:
    """
    From Dino, feature extraction utils
    Load a lookup dictionary, mapping string to rank, from a CSV file.
    Assumes the CSV to be pre-sorted.
    :param csv_source: Source data filepath
    :param count: number of strings to load
    :return: dictionary mapping strings to rank
    """
    lookup_dict: Dict[str, int] = dict()
    rank: int = 0
    with open(csv_source, 'r') as fd:
        line = next(fd)
        try:
            while rank < count:
                if line.startswith("#"):
                    line = next(fd)
                    continue
                lookup_dict[line.strip().split(',')[-1]] = rank
                rank += 1
                line = next(fd)
        except StopIteration:
            raise RuntimeError(f"Not enough entries in file. Expected at least {count}, max is {rank}.")

    return lookup_dict

def url_parse_to_uniform(url: str) -> str:
    """takes a url (incl. attributes etc) and returns a uniform format"""
    obj = parse.urlsplit(url)
    #remove leading www. etc and add to url set
    return url_to_uniform_domain(obj.netloc)

def url_to_uniform_domain(url: str) -> str:
    """
    Takes a URL or a domain string and transforms it into a uniform format.
    Examples: {"www.example.com", "https://example.com/", ".example.com"} --> "example.com"
    :param url: URL to clean and bring into uniform format
    """
    new_url = url.strip()
    new_url = re.sub("^http(s)?://", "", new_url)
    new_url = re.sub("^www([0-9])?", "", new_url)
    new_url = re.sub("^\\.", "", new_url)
    new_url = re.sub("/$", "", new_url)
    return new_url

def main() -> None:
    argv=None
    cargs = docopt(__doc__, argv=argv)

    dat_path: str = cargs["<data_path>"]
    in_pixels: Dict[str, Dict[str, Any]] = dict()
    out_pixels: Dict[str, Dict[str, Any]] = dict()
    top_domain = load_lookup_from_csv("../resources/top_domains.csv", 500)
    c_changes = [[0, 0], [0, 0]]
    
    if not os.path.exists(dat_path):
        print(f"{dat_path} is not a valid data path")
        return

    with open(dat_path) as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")
    
    gnl = 0
    matched = 0

    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        if cat == 0 or cat == 1: #only relabel samples from necessary and functional category
            pixel_domain: str = url_parse_to_uniform(sample["url"])
            if pixel_domain in top_domain:
                rank = top_domain[pixel_domain]
                m = sample["matched"]
                if (rank in [0, 25, 110]): #google-analytics.com, ssl.google-analytics.com, stats.g.doubleclick.net
                    sample["label"] = 2
                    sample["moved"] = 1
                    c_changes[cat][0] += 1
                    matched += m
                elif rank in [1, 5, 9, 10]: #facebook.com, cm.g.doubleclick.net, track.hubspot.com, pagead2.googlesyndication.com
                    sample["label"] = 3
                    c_changes[cat][1] += 1
                    sample["moved"] = 1
                    matched+=m
                elif (rank == 2 or rank == 3) and ("pagead/1p-conversion" in sample["url"] or "ads/ga-audiences" in sample["url"] or "pagead/1p-user-list" in sample["url"] or "pagead/landing" in sample["url"]):
                    #pagead/1p-conversion and ad/ga-audiences etc from google.com and google.nl moved to cat 3
                    sample["label"] = 3
                    c_changes[cat][1] += 1
                    sample["moved"]=1
                    matched+=m
                    if rank==3:
                        gnl += 1
            out_pixels[key] = sample

        else:
            out_pixels[key] = sample
    
    print(f"Nr of samples returned: {len(out_pixels)}")
    print(f"Nr of necessary samples moved to another category: {sum(c_changes[0])}: to analytics: {c_changes[0][0]}, to advertising: {c_changes[0][1]}")
    print(f"Nr of functional samples moved to another category: {sum(c_changes[1])}: to analytics: {c_changes[1][0]}, to advertising: {c_changes[1][1]}")
    print(f"Nr google.nl moved: {gnl}")
    print(f"Nr of matched samples moved: {matched}")

    """
    out1 = os.path.basename(dat_path).split('.')[0]
    out_name = str(out1) + 'reclassified.json'
    with open(out_name, "w") as outf:
        json.dump(out_pixels, outf)
    print(f"reclassified samples returned in {out_name}")
    """

if __name__ == "__main__":
    exit(main())

