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

    count1 = 0
    count2 = 0
    count3 = 0
    pink_count = 0
    adobe = 0
    o7 = 0 

    for key in in_pixels:
        sample = in_pixels[key]
        if sample["matched"] == 1 and (sample["img_size"][0] > 1 or sample["img_size"][1] > 1):
            #print(sample)
            count1 += 1
            if "pinkgellac.nl" in sample["url"]:
                pink_count += 1
        elif sample["matched"] == 0 and (sample["blocked"][0] == 1 or sample["blocked"][1] == 1) and (sample["img_size"][0] == 2 and sample["img_size"][1] == 2):
            url = sample["url"]
            if "omtrdc.net" in url:
                adobe += 1
            elif "2o7.net" in url:
                o7 += 1
            #else:
                #print(sample)
            count2 += 1
        elif sample["matched"] == 0 and (sample["img_size"][0] == 1 and sample["img_size"][1] == 1) and (sample["blocked"][0] == 0 and sample["blocked"][1] == 0):
            print(sample)
            count3 += 1

    print(f"matched large images: {count1}")
    print(f"pinkgellac.nl count: {pink_count}")
    print(f"unmatched, blocked 2x2 images: {count2}")
    print(f"adobe: {adobe}, 07: {o7}")
    print(F"unmatched 1x1 not blocked: {count3}")

if __name__ == "__main__":
    exit(main())

