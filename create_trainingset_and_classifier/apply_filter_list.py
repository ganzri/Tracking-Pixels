"""
calculates how many training samples are per category
list, which easylist to use



Usage:
    cat_stats.py <data_path> <EasyPrivacy_path> <EasyList_path>
"""

import json
from typing import List, Dict, Any, Tuple
from collections import Counter
from docopt import docopt
import os
from abp_blocklist_parser import BlockListParser
from urllib import parse
import re

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

def is_third_party(url, first_party_domain) -> bool:
    pixel_domain = url_to_uniform_domain(parse.urlsplit(url).netloc)
    website_domain = url_to_uniform_domain(parse.urlsplit(first_party_domain).netloc)
    return (pixel_domain not in website_domain)

def main() -> None:
    argv = None
    cargs = docopt(__doc__, argv=argv)
    
    dat_path: str = cargs["<data_path>"]
    EasyPrivacy: str = cargs["<EasyPrivacy_path>"]
    EasyList: str = cargs["<EasyList_path>"]

    if not os.path.exists(dat_path):
        print(f"{dat_path} is not a valid data path")
        return 
    if not os.path.exists(EasyPrivacy):
        print(f"{EasyPrivacy} is not a valid path to the list")
        return
    if not os.path.exists(EasyList):
        print(f"{EasyList} does not exist")
        return

    in_pixels: Dict[str, Dict[str, Any]] = dict()
    out_pixels: Dict[str, Dict[str, Any]] = dict()
    with open(dat_path) as fd: 
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")

    blocklist_parser = BlockListParser(EasyPrivacy)
    blocklist_parser_2 = BlockListParser(EasyList)
    options = dict()
    options["image"] = 1
    #third party is set later. Other options??


    category = ["necessary", "functional", "analytics", "advertising"]
    category_count = [0] * 4
    cat_blocked_tr_list = [0] * 4
    cat_blocked_new_list = [0] * 4
    cat_blocked_list_2 = [0] * 4
    blocked_tot = [0] * 4

    
    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        category_count[cat] += 1
        bl = 0
        dom = url_to_uniform_domain(parse.urlsplit(sample["triggering_origin"]).netloc)
        options["domain"] = dom
        filters = [0]*2 #0:EasyPrivacy, 1:EasyList
        if sample["blocked"] == 1:
            cat_blocked_tr_list[cat] += 1
            bl = 1
        if is_third_party(sample["url"], sample["first_party_domain"]):
            options["third-party"] = 1
        else:
            options["third-party"] = 0
        if blocklist_parser.should_block(sample["url"], options):    
            cat_blocked_new_list[cat] += 1
            filters[0] = 1
            bl = 1
        if blocklist_parser_2.should_block(sample["url"], options):
            cat_blocked_list_2[cat] += 1
            bl = 1
            filters[1] = 1
        blocked_tot[cat] += bl
        sample["blocked"] = filters
        out_pixels[key] = sample
        """
        if cat == 1 and bl == 1:
            print(sample["url"])
            print(blocklist_parser.should_block_with_items(sample["url"], options))
            print(blocklist_parser_2.should_block_with_items(sample["url"], options))
        """
    print(f"{len(out_pixels)} samples returned")
    json_outfile = "./training_data_output_offline_model/10_12_2021_filter_lists.json"
    with open(json_outfile, 'w') as fd:
        json.dump(out_pixels, fd)

    for i in range(4):
        print(f"{category[i]}: Nr of samples: {category_count[i]}, Nr blocked by tracking server blocklist: {cat_blocked_tr_list[i]}, Nr blocked by new list: {cat_blocked_new_list[i]}, Nr blocked list 2: {cat_blocked_list_2[i]}, total blocked: {blocked_tot[i]}")
    

if __name__ == "__main__":
    exit(main())
