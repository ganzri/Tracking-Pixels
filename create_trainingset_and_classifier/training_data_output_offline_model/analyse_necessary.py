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
from collections import Counter
import numpy
import re
from urllib import parse
from math import log
import matplotlib.pyplot as plt

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
    out_pixels: Dict[str, Dict[str, Any]] = dict()
    top_domain = load_lookup_from_csv("../resources/top_domains.csv", 500)
    url_entropy = []
    url_entropy_n_1x1 = []
    url_entropy_n = []
    for i in range(4):
        url_entropy.append([])


    with open('first_filter.json') as fd:
        in_pixels = json.load(fd)
    print(f"Nr of samples loaded: {len(in_pixels)}")
    
    for key in in_pixels:
        sample = in_pixels[key]
        cat = sample["label"]
        if 0<= cat <= 3:
            url_se = shannon_entropy(sample["url"])
            url_entropy[cat].append(url_se)
            if cat == 0 and sample["img_size"][0] == 1 and sample["img_size"][1] == 1:
                url_entropy_n_1x1.append(url_se)
                out_pixels[key] = sample
            elif cat == 0:
                url_entropy_n.append(url_se)

    plt.hist(url_entropy[0], 100, label='Necessary')
    plt.hist(url_entropy[3], 100, label='Advertising')
    plt.hist(url_entropy[2], 100, label='Analytics')
    plt.hist(url_entropy[1], 100, label='Functional')
    plt.legend()
    plt.title("Histogram of Shannon Entropy in URL per category")
    plt.savefig('Shannon_Entropy_URL_per_cat.png')
    plt.clf()
    
    plt.hist(url_entropy_n, 100, label = "Necessary non 1x1")
    plt.hist(url_entropy_n_1x1, 100, label = "Necessary 1x1")
    plt.legend()
    plt.title("Shannon Entropy URL for Necessary")
    plt.savefig('Shannon_Entropy_Necessary.png')
    plt.clf()
    print(f"Nr of samples returned: {len(out_pixels)}")

    with open("necessary_1x1.json", "w") as outf:
        json.dump(out_pixels, outf)
    
    nec_1x1: Dict[str, Dict[str, Any]] = dict()
    top_domain_1000 = load_lookup_from_csv("../resources/top_domains.csv", 1000)

    with open("necessary_1x1.json") as fd:
        nec_1x1 = json.load(fd)
    print(f"Nr of samples loaded: {len(nec_1x1)}")
    rank_count = [0] * 1000
    
    for key in nec_1x1:
        sample = nec_1x1[key]
        url_se = shannon_entropy(sample["url"])
        if url_se > 5:
            #check if these high entropy urls are all from the same few domains
            pixel_domain: str = url_parse_to_uniform(sample["url"])
            if pixel_domain in top_domain:
                rank = top_domain[pixel_domain]
                rank_count[rank] += 1
    for i in range(1000):
        if rank_count[i] != 0:
            print(f"domain nr {i}, count {rank_count[i]}") 


if __name__ == "__main__":
    exit(main())

