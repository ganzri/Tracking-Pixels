"""
Input JSON structure:
    [
     "pixel_id_1": {
            "visit_id": "<visit_id>",
            "request_id": "<request_id>",
            "name": "<name>",
            "url": "<url>",
            "first_party_domain": "<site_url>",
            "label": [0-3],
            "triggering_origin": "<triggering_origin>",
            "headers": "<headers>",
            "img_format": img_data[0],
            "img_size": "(width, height)"
            "img_mode": img_data[2],
            "img_colour": "(r,g,b,alpha)",
            "id": "<id>,
            "matched": "1/0",
            "moved": "1/0"
      },
      "pixel_id_2": {
      ...
      },
      ...
    ]

Usage:
    noise_analysis.py <data_path>
"""

import os
import json
from typing import List, Dict, Any, Tuple
from docopt import docopt
import re
from urllib import parse
from numpy import sort

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
    
    distinct_pixels = set()
    to_analyse = 0
    
    with open("necessary_to_check.txt", 'w') as fd:
        fd.write("visit_id request_id url")
        for k in in_pixels:
            sample = in_pixels[k]
            purpose = sample["label"]
            url = sample["url"]
            matched = sample["matched"]
            if matched == 1 and purpose == 0:
                u_url = url_parse_to_uniform(url)
                path = parse.urlsplit(url).path
                key = u_url+path
                if key not in distinct_pixels:
                    distinct_pixels.add(key)
                    fd.write(f"{sample['visit_id']} {sample['request_id']} {url} \n")
                    to_analyse += 1

    print(f"nr of samples in necessary to analyse: {to_analyse}")
if __name__ == "__main__":
    exit(main())

