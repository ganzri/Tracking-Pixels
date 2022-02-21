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

mode is one of query, path 
query is the value we want to ask about

Usage:
    domain_to_query.py <mode> <query>
"""

import json
from typing import Dict, Any, List
from urllib import parse
from docopt import docopt

def main() -> None:
    argv = None
    cargs = docopt(__doc__, argv=argv)

    mode: str = cargs["<mode>"]
    to_query: str = cargs["<query>"]

    in_pixels: Dict[str, Dict[str, Any]] = dict()

    with open('10_12_2021_13_40.json') as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")
    
    domains = dict()

    for key in in_pixels:
        sample = in_pixels[key]
        url = sample["url"]
        u = parse.urlsplit(url)
        if mode == "query":
            q_dict = parse.parse_qs(u.query)
            if to_query in q_dict:
                if u.netloc in domains:
                    domains[u.netloc] += 1
                else:
                    domains[u.netloc] = 1
        if mode == "path":
            path = u.path.split("/")
            if to_query in path:
                if u.netloc in domains:
                    domains[u.netloc]+=1
                else:
                    domains[u.netloc] = 1

    print(f"{mode} {to_query} was found in: ")
    print(domains)
    print(f"nr of netlocs using this query param: {len(domains)}")

if __name__ == "__main__":
    exit(main())

