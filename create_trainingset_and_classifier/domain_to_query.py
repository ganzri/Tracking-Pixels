# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
This script returns an overview of netlocs that use either a path piece or a query parameter, and how often they occur in the data set.
This was used to assess how general the important features are.
for example path piece "collect" is used by Google-Analytics and around 50 other netlocs.

mode is one of query (for query parameters), or path (for path pieces)
query is the value we want to ask about (a string)

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

