# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Similar to analyse but to be used with the larger (200, or all) data set, but because track.hubspot.com and some others are very, very common in this data set, they are filtered out. This is just for convenience as they have been dealt with in the 20 samples set.
Prints the samples in the json to the console, to be manually analysed. Adds details on the filter list decision (which list would block it, based on what rule)

additionally compares whether the two data sets differe (this is a very rough estimate of their differing)

<gt_data> the actual data one wants to analyse: ground truth vs blocked or not
<pred_data> the same comparison but predicted vs blocked or not
e.g. if gt_data is gt_aa_not_blocked.json then pred_data should be pred_aa_not_blocked.json

Usage:
    analyse.py <gt_data> <pred_data>
"""

import json
from typing import List, Dict, Any, Tuple
from docopt import docopt
import os
from abp_blocklist_parser import BlockListParser
from urllib import parse
import re

def is_third_party(url, first_party_domain) -> bool:
    pixel_domain = url_to_uniform_domain(parse.urlsplit(url).netloc)
    website_domain = url_to_uniform_domain(parse.urlsplit(first_party_domain).netloc)
    return (pixel_domain not in website_domain)

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
    argv = None
    cargs = docopt(__doc__, argv=argv)

    dat_path: str = cargs["<gt_data>"]
    dat_path_2: str = cargs["<pred_data>"]

    if not os.path.exists(dat_path): 
        print(f"datapath is not a valid path")
        return

    if not os.path.exists(dat_path_2):
        print(f"datapath is not a valid path")
        return

    gt: Dict[str, Dict[str, Any]] = dict()
    pred: Dict[str, Dict[str, Any]] = dict()
    with open(dat_path) as fd:
        gt = json.load(fd)
    print(f"Nr of gt samples loaded: {len(gt)}")
    with open(dat_path_2) as fd:
        pred = json.load(fd)
    print(f"Nr of pred samples loaded: {len(pred)}")


    a = set()
    b = set()
    for k in gt:
        a.add(k)
    for k in pred:
        b.add(k)

    print(len(a-b))
    print(len(b-a))
    print(len(a))
    print(len(b))
    
    """
    print("only in gt")
    keys_gt_only = a-b
    for k in keys_gt_only:
        print(gt[k])
        print("\n")
    """
    """
    print("only in pred")
    keys_in_pred_only = b-a
    for k in keys_in_pred_only:
        print(pred[k])
        print("\n")
    
    """    
    privacy = "./abp_blocklist_parser/easyprivacy.txt"
    easylist = "./abp_blocklist_parser/whole_list.txt"
    blocklist_parser = BlockListParser(privacy)
    blocklist_parser_2 = BlockListParser(easylist)
    options = dict()
    options["image"] = 1

    
    #used to count and exclude hubspot
    i = 1
    hub = 0
    outbr = 0
    for k in gt:
        #print(k)
        sample = gt[k]
        url = sample["url"]
        if "track.hubspot.com" in url:
            hub += 1
        elif "tr.outbrain.com" in url:
            outbr += 1
        else:
            dom = url_to_uniform_domain(parse.urlsplit(sample["triggering_origin"]).netloc)
            options["domain"] = dom
            if is_third_party(sample["url"], sample["first_party_domain"]):
                options["third-party"] = 1
            else:
                options["third-party"] = 0
    
            print(i)
            i += 1
            print(sample)
            print(blocklist_parser.should_block_with_items(sample["url"], options))
            print(blocklist_parser_2.should_block_with_items(sample["url"],options))
            print(sample["matched"])
            print(sample["img_size"])
            print("\n")
    print(f"hubspot found {hub} times")
    print(f"outbrain found {outbr} times")


if __name__ == "__main__":
    exit(main())
                    
