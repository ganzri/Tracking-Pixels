"""
Script to estimate the noise in the dataset based on deviation from majority class.

Input are json files.
original_data refers to the .json before necessary and functional category was filtered and trackers from 9 netlocs were moved to 
appropriate categories. This is only used to assess the noise in the consent declarations before we do the "clean-up". We only consider the samples matched to a consent declaration in this dataset.

moved_data is after. This is the data that was used for training the model. The noise is calculated over the whole dataset, i.e.,
including images.

Usage:
    noise_analysis.py <original_data> <moved_data>
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

    original: str = cargs["<original_data>"]
    if not os.path.exists(original):
        print(f"{original} is not a valid data path")
        return

    moved: str = cargs["<moved_data>"]
    if not os.path.exists(moved):
        print(f"{moved} is not a valid data path")

    original_pixels: Dict[str, Dict[str, Any]] = dict()
    moved_pixels: Dict[str, Dict[str, Any]] = dict()

    with open(original) as fd:
        original_pixels = json.load(fd)
    print(f"Nr of samples loaded original data: {len(original_pixels)}")
    
    with open(moved) as fd:
        moved_pixels = json.load(fd)
    print(f"Nr of samples loaded moved data: {len(moved_pixels)}")
    print("\n")   

    #first only look at the original matches between categories from consent notices and observed pixels
    #i.e., this gives an indication of the noise in the observed consent notices that we could match and not in the whole dataset
    distinct_pixels = set()
    majority_class: Dict[str, [4]] = dict()
    nr_declared_pixels = 0

    for k in original_pixels:
        sample = original_pixels[k]
        matched = sample["matched"]
        if matched == 1: #we only look at matched pixels
            purpose = sample["label"]
            url  = sample["url"]
            nr_declared_pixels += 1
            u_url = url_parse_to_uniform(url)
            path = parse.urlsplit(url).path
            key = u_url+path
            distinct_pixels.add(key)
            if key in majority_class:
                majority_class[key][purpose] +=1
            else:
                majority_class[key] = [0, 0, 0, 0]
                majority_class[key][purpose] = 1

    distinct_third_party = 0
    nr_occurence_third_party = 0
    deviation_from_majority = 0

    for key in majority_class:
        if sum(majority_class[key]) > 1:
            distinct_third_party +=1
            nr_occurence_third_party += sum(majority_class[key])
            s = sort(majority_class[key])
            deviation_from_majority += sum(s[:3])

    #how many unique pixels that occur more than once
    print(f"Noise analysis of observed pixels matched to consent notices")
    print(f"nr of declared pixels: d = {nr_declared_pixels}")
    print(f"unique third party pixels matched: {distinct_third_party}")
    print(f"total nr of unique pixels matched: {len(distinct_pixels)}")
    print(f"declared third party pixels matched (can match the same pixel multiple times): a = {nr_occurence_third_party}")
    print(f"nr of matches deviating from majority class: b = {deviation_from_majority}")
    print(f"lower bound for noise in third party pixels: (b / a ) = {deviation_from_majority/nr_occurence_third_party}")
    print(f"lower bound for noise all pixels: b / d = {deviation_from_majority/nr_declared_pixels}")
    print("\n")

    #noise per category
    maj_m = [0,0,0,0]
    tot_m = [0,0,0,0]

    for k in original_pixels:
        sample = original_pixels[k]
        matched = sample["matched"]
        if matched == 1:
            purpose = sample["label"]
            url = sample["url"]
            u_url = url_parse_to_uniform(url)
            path = parse.urlsplit(url).path
            key = u_url+path
            a = majority_class[key]
            index_max = max(range(len(a)), key=a.__getitem__)
            if index_max == purpose: #we are in the majority class
                maj_m[purpose] += 1
            tot_m[purpose] += 1

    tdev = 0
    for i in range(4):
        dev = tot_m[i] - maj_m[i]
        per = dev/tot_m[i]
        tdev+=dev
        print(f"{i}: nr of samples: {tot_m[i]}, deviating from majority class: {dev} percent {per}")
    print(f"total nr of deviations {tdev}")
    print("\n")

    ###############################################################################################
    #Noise analysis for the whole dataset after adding to necessary and moving some pixels based
    distinct_pixels = set()
    majority_class: Dict[str, [4]] = dict()
    nr_pixels = len(moved_pixels)
    
    for k in moved_pixels:
        sample = moved_pixels[k]
        purpose = sample["label"]
        url = sample["url"]
        u_url = url_parse_to_uniform(url)
        path = parse.urlsplit(url).path
        key = u_url+path
        distinct_pixels.add(key)
        if key in majority_class:
            majority_class[key][purpose] +=1
        else:
            majority_class[key] = [0, 0, 0, 0]
            majority_class[key][purpose] = 1

    distinct_third_party = 0
    nr_occurence_third_party = 0
    deviation_from_majority = 0

    for key in majority_class:
        if sum(majority_class[key]) > 1:
            distinct_third_party +=1
            nr_occurence_third_party += sum(majority_class[key])
            s = sort(majority_class[key])
            deviation_from_majority += sum(s[:3])

    #how many unique pixels that occur more than once
    print(f"Noise analysis of observed pixels after adding to necessary category and moving samples from 9 netlocs")
    print(f"nr of pixels and images: d = {nr_pixels}")
    print(f"unique third party pixels/ images: {distinct_third_party}")
    print(f"total nr of unique pixels/ images: {len(distinct_pixels)}")
    print(f"How many samples (matches) come from third party pixels/ images: a = {nr_occurence_third_party}")
    print(f"nr of matches deviating from majority class: b = {deviation_from_majority}")
    print(f"lower bound for noise in third party pixels: (b / a ) = {deviation_from_majority/nr_occurence_third_party}")
    print(f"lower bound for noise all pixels: b / d = {deviation_from_majority/nr_pixels}")
    print("\n")

    #noise per category
    maj_m = [0,0,0,0]
    tot_m = [0,0,0,0]

    for k in moved_pixels:
        sample = moved_pixels[k]
        purpose = sample["label"]
        url = sample["url"]
        u_url = url_parse_to_uniform(url)
        path = parse.urlsplit(url).path
        key = u_url+path
        a = majority_class[key]
        index_max = max(range(len(a)), key=a.__getitem__)
        if index_max == purpose: #we are in the majority class
            maj_m[purpose] += 1
        tot_m[purpose] += 1

    tdev = 0
    for i in range(4):
        dev = tot_m[i] - maj_m[i]
        per = dev/tot_m[i]
        tdev+=dev
        print(f"{i}: nr of samples: {tot_m[i]}, deviating from majority class: {dev} percent {per}")
    print(f"total nr of deviations {tdev}")

if __name__ == "__main__":
    exit(main())

