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
    out_pixels: Dict[str, Dict[str, Any]] = dict()

    count1 = 0
    for key in in_pixels:
        sample = in_pixels[key]
        count1 += 1
        if count1 < 10:
            out_pixels[key] = sample

    print(f"Nr of samples returned: {len(out_pixels)}")
    with open("ten.json", "w") as outf:
        json.dump(out_pixels, outf)
if __name__ == "__main__":
    exit(main())

