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
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from collections import Counter
from math import log

def shannon_entropy(text: str) -> float:
    """
    same definition as in feature processor for feature extraction
    calculates shannon entropy of a given string
    """
    content_char_counts = Counter([ch for ch in text])
    total_string_size = len(text)
    entropy: float = 0
    for ratio in [char_count / total_string_size for char_count in content_char_counts.values()]:
        entropy -= ratio * log(ratio, 2)
    return entropy

def main() -> None:
    in_pixels: Dict[str, Dict[str, Any]] = dict()

    with open('10_12_2021_13_40.json') as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")
    
    header_fields: Dict[str, int] = dict()
    colour = []
    url_entropy = []
    headers_entropy = []
    transparency = []

    for i in range(4):
        colour.append([])
        url_entropy.append([])
        headers_entropy.append([])
        transparency.append([])

    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        #collect header fields and count how often each header field is found
        headers = sample["headers"]
        h2 = headers[2:-2]
        h3 = h2.split("],[")
        for field in h3:
            field_name = field.split('","')[0][1:]
            if field_name in header_fields:
                header_fields[field_name] += 1
            else:
                header_fields[field_name] = 1
        #histogramms for colour, entropy and transparency
        col = (sample["img_colour"][0] + sample["img_colour"][1] + sample["img_colour"][2])/3
        colour[cat].append(col)
        url_se = shannon_entropy(sample["url"])
        headers_se = shannon_entropy(headers)
        url_entropy[cat].append(url_se)
        headers_entropy[cat].append(headers_se)
        transparency[cat].append(sample["img_colour"][3])
        
    print(header_fields)
    plt.hist(colour[0], 100, label='Necessary')
    plt.hist(colour[1], 100, label='Functional')
    plt.hist(colour[2], 100, label='Analytics')
    plt.hist(colour[3], 100, label='Advertising')
    plt.legend()
    plt.title("Distribution of avg. colour value (r + g + b) / 3")
    plt.savefig('colour.png')
    plt.clf()

    plt.hist(url_entropy[0], 100, label='Necessary', color=(4/255, 108/255, 212/255, 0.7))
    plt.hist(url_entropy[1], 100, label='Functional', color=(212/255, 139/255, 4/255, 0.7))
    plt.hist(url_entropy[2], 100, label='Analytics', color=(35/255, 138/255, 7/255, 0.7))
    plt.hist(url_entropy[3], 100, label='Advertising', color=(184/255, 27/255, 13/255, 0.7))
    plt.legend()
    plt.title("Distribution of Shannon entropy in URL")
    plt.savefig('Shannon_Entropy_URL_per_cat.png')
    plt.clf()
   
    plt.hist(headers_entropy[0], 100, label='Necessary')
    plt.hist(headers_entropy[1], 100, label='Functional')
    plt.hist(headers_entropy[2], 100, label='Analytics')
    plt.hist(headers_entropy[3], 100, label='Advertising')
    plt.legend()
    plt.title("Distribution of Shannon entropy in headers")
    plt.savefig('Shannon_Entropy_Headers_per_cat.png')
    plt.clf()
    
    plt.hist(transparency[0], 100, label='Necessary')
    plt.hist(transparency[1], 100, label='Functional')
    plt.hist(transparency[2], 100, label='Analytics')
    plt.hist(transparency[3], 100, label='Advertising')
    plt.legend()
    plt.title("Distribution of transparency")
    plt.savefig('transparency.png')

if __name__ == "__main__":
    exit(main())

