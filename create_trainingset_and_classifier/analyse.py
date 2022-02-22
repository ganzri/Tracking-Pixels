# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Prints the samples in the json stored at <data_path> to the console, to be manually analysed. Adds details on the filter list decision (which list would block it, based on what rule)
Usage:
    analyse.py <data_path> 
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

    dat_path: str = cargs["<data_path>"]

    if not os.path.exists(dat_path):
        print(f"{dat_path} is not a valid data path")
        return

    in_pixels: Dict[str, Dict[str, Any]] = dict()
    with open(dat_path) as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")    

    privacy = "./abp_blocklist_parser/easyprivacy.txt"
    easylist = "./abp_blocklist_parser/whole_list.txt"
    blocklist_parser = BlockListParser(privacy)
    blocklist_parser_2 = BlockListParser(easylist)
    options = dict()
    options["image"] = 1

    i = 1
    for k in in_pixels:
        print(i) #for convenience, this nr has not other significance
        i += 1
        print(k) #global key of this sample
        print(in_pixels[k])
        print("\n")
        sample = in_pixels[k]
        dom = url_to_uniform_domain(parse.urlsplit(sample["triggering_origin"]).netloc)
        options["domain"] = dom
        if is_third_party(sample["url"], sample["first_party_domain"]):
            options["third-party"] = 1
        else:
            options["third-party"] = 0
        print(blocklist_parser.should_block_with_items(sample["url"], options))
        print(blocklist_parser_2.should_block_with_items(sample["url"],options))
        print(sample["matched"])
        print(sample["img_size"])
        print("\n")
if __name__ == "__main__":
    exit(main())
                    
