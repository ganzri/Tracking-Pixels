#TODO exclude filter list or include correct version
"""
Matches pixels fount in http_requests and responses to their labels found in the consent notice (all from tables in <db_path> db) 
based on
1. name of pixel found in consent notice can be found in the path or netloc of the url of the http_request for an image resource
2. domain of pixel found in consent notice matches netloc of url in http_request
3. request_id the same in http_request and response

1+2 are already implemented in create_trainingset.py for the online model. To save time, I import its output as <input_data> here. 

In addition, all unmatched images (i.e. images without consent declaration) are put into the necessary category, and their 
http_request and response are linked by having the same (visit_id, request_id) pair.

Using the content_hash from the http request, information (size, colour ...)  about the image requested is extracted from a leveldb 
at <ldb_path>

outputs the labeled pixel data to a JSON format. Each observed pixel is an object with attributes inside the JSON. The JSON is then
used for feature extraction in the classifier.
"""
#TODO updata JSON structur
"""
Output JSON structure:
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
            "img_colour": "(r,g,b,alpha)"

      }, 
      "pixel_id_2": {
      ...
      }, 
      ...
    ]

also conducts some statistics, that could not be performed in "post_process.py" as the matching of pixels requires more functionality
than SQL-queries provided.

Usage:
    create_trainingset_offline_model.py <db_path> <ldb_path> <input_data>

Options:
    -h --help       Show this help message
"""

from numpy import sort
from urllib import parse
import sqlite3
import re
import os
import json
import traceback
import logging
import random

from statistics import mean, stdev
from datetime import datetime
from typing import List, Dict, Any, Tuple
from docopt import docopt

import plyvel
from PIL import Image
import io
import PIL
import warnings
from abp_blocklist_parser import BlockListParser

logger = logging.getLogger("feature-extract")
time_format = "%Y-%m-%dT%H:%M:%S.%fZ"

def setupLogger(logpath: str = "./training_data_output_offline_model_v2/extract_pixel_data_offline_model.log") -> None:
    """
    Set up the logger instance
    Code from Dino extract_cookie_data_from_db.py
    """
    with open(logpath, 'w') as fd:
        pass

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    fh = logging.FileHandler(logpath)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def analyse_image(content_hash: str, db: plyvel.DB, img_data: tuple) -> bool:
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    warnings.simplefilter('error', UserWarning)
    hash_bytes = str.encode(content_hash)
    img = db.get(hash_bytes)
    try:
        image = Image.open(io.BytesIO(img)) #images are stored as byte object in leveldb, hence import them this way
        # there are other options for import in doc for PIL image, maybe change later
        #attributes of Image class
        filename = image.filename # probably always empty
        image_format = image.format #gave GIF for my pixel
        size = image.size #or .width and .height
        mode = image.mode # "1", "L", "RGB", "CMYK"
        palette = image.palette #optional attribute only if mode is P or PA
        info = image.info #dict holding data associated with image
        # anim = image.is_animated #does not work for jpeg
        # n_frames = image.n_frames #dito

        #get colour of first pixel, maybe could do this separately for each format instead of casting to RGBA
        rgb_img = image.convert("RGBA")
        colour = rgb_img.getpixel((0,0))
        #higher alpha channel values == more opaque, (255, 255 255) is white
        
        img_data.append(image_format)
        img_data.append(size)
        img_data.append(mode)
        img_data.append(colour)
        
    except (PIL.UnidentifiedImageError, OSError, ValueError, UserWarning, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return False
    return True

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

def is_third_party(url, first_party_domain) -> bool:
    pixel_domain = url_to_uniform_domain(parse.urlsplit(url).netloc)
    website_domain = url_to_uniform_domain(parse.urlsplit(first_party_domain).netloc)
    return (pixel_domain not in website_domain)


def main()->int:
    argv = None
    cargs = docopt(__doc__, argv=argv)

    random.seed(299756)

    setupLogger()
    
    database_path = cargs["<db_path>"]
    ldb_path = cargs["<ldb_path>"]
    input_path = cargs["<input_data>"]

    if not os.path.exists(database_path) and os.path.exists(ldb_path) and os.path.exists(input_path):
        logger.error("Database file does not exist.")
        return 1
    
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row #enables dictionary access by column name

    logger.info("Begin training data extraction")
    
    #load training data from online model, we can use this to then link it to the response and the image features
    with open(input_path) as js:
        json_from_online: Dict[str, Dict[str, Any]] = json.load(js)
    logger.info(f"loaded {len(json_from_online)} samples from online model")

    json_training_data_pixels: Dict[str, Dict[str, Any]] = dict()
    
    #setup to be able to query the easyprivacy trackingserver list if a url is blocked for an image resource
    blocklist_parser = BlockListParser("./abp_blocklist_parser/easyprivacy_trackingservers.txt")
    options = dict()
    options["image"] = 1
    #third party is set later. Other options??

    try:
        with conn:
            logger.info("start json for pixels matched to declaration")
            potential_pixels = 0 #count all images of size 1x1
            potential_images = 0 #count larger images
            error_count = 0 #count nr of errors due to not being able to read image data

            #dict to query content_hash (because SQL queries are slow, creating dict once hopefully speeds up things)
            #for all json and dict I use visit_id + "_" + request_id as key, e.g. 1234_223, as this uniquely identifies a
            #request/response for a given resource (image)
            hash_query: Dict[str, Any] = dict()
            """
            cur = conn.execute("SELECT * FROM http_responses;")
            for row in cur:
                key = str(row["visit_id"]) + "_" + str(row["request_id"])
                hash_query[key] = row["content_hash"]
            logger.info(f"nr of http_response entries: {len(hash_query)}")
            with open("has_query.json", 'w') as fd:
                json.dump(hash_query, fd)
            
            cur.close()
            """
            with open("has_query.json") as fd:
                hash_query = json.load(fd)
            logger.info(f"size of hash_query: {len(hash_query)}")

            #DB to query image parameters
            db = plyvel.DB(ldb_path, create_if_missing=False)
            
            # add all images that could be matched to a consent declaration. Add if an image is returned in the http_response and
            # if that image does not cause an error in analyse_image
            for key in json_from_online:
                sample = json_from_online[key]
                vi = sample["visit_id"]
                ri = sample["request_id"]
                key = str(vi) + "_" + str(ri)
                try:
                    content_hash = hash_query[key]
                    
                    img_data = []
                    if  analyse_image(content_hash, db, img_data):
                        if is_third_party(sample["url"], sample["first_party_domain"]):
                            options["third-party"] = 1
                        else:
                            options["third-party"] = 0

                        if blocklist_parser.should_block(sample["url"], options):
                            blocked = 1
                        else:
                            blocked = 0

                        json_pixel_key = str(sample["visit_id"]) + "_" + sample["request_id"]
                
                        json_training_data_pixels[json_pixel_key] = {
                            "visit_id": sample["visit_id"],
                            "request_id": sample["request_id"],
                            "url": sample["url"],
                            "first_party_domain": sample["first_party_domain"],
                            "label": sample["label"],
                            "triggering_origin": sample["triggering_origin"],
                            "headers": sample["headers"],
                            "img_format": img_data[0],
                            "img_size": img_data[1],
                            "img_mode": img_data[2],
                            "img_colour": img_data[3],
                            "id": random.random(),
                            "matched": 1,
                            "moved": 0,
                            "blocked": blocked
                        }
                        if img_data[1][0] == 1 and img_data[1][1] == 1:
                            potential_pixels += 1
                        else:
                            potential_images += 1

                    else:
                        error_count += 1
           
                    
                except KeyError:
                    continue
            
            nr_md = len(json_training_data_pixels)
            logger.info(f"extracted {nr_md} training records from pixels matched to consent declarations")
            logger.info(f"of these {potential_pixels} are 1x1 and {potential_images} are larger")
            logger.info(f"nr of errors: {error_count}")

            logger.info("start extracting training samples from unmatched images")
            error_count = 0
            #put all unmatched images into categorie necessary = 0
            #all images that where not matched to a consent declaration, i.e. images not yet in the json_training_data_pixels,
            #are added as above but with cat_id = 0
            cur = conn.cursor()
            cur.execute("SELECT * FROM view_first_request_images c JOIN site_visits s ON s.visit_id == c.visit_id;")
            for row in cur:
                key = str(row["visit_id"]) + "_"   + str(row["request_id"])
                if key not in json_training_data_pixels:
                    cat_id = 0
                    try:
                        content_hash = hash_query[key]
                        # get information about the image from the leveldb
                        # to put into json, if an error in leveldb occured, sample is discarded
                        img_data = []
                        if  analyse_image(content_hash, db, img_data):
                            if is_third_party(row["url"], row["site_url"]):
                                options["third-party"] = 1
                            else:
                                options["third-party"] = 0

                            if blocklist_parser.should_block(row["url"], options):
                                blocked = 1
                            else:
                                blocked = 0

                            if img_data[1][0] == 1 and img_data[1][1] == 1:
                                potential_pixels += 1
                            else:
                                potential_images += 1
                                
                                
                            json_training_data_pixels[key] = {
                                "visit_id": row["visit_id"],
                                "request_id": row["request_id"],
                                "url": row["url"],
                                "first_party_domain": row["site_url"],
                                "label": cat_id,
                                "triggering_origin": row["triggering_origin"],
                                "headers": row["headers"],
                                "img_format": img_data[0],
                                "img_size": img_data[1],
                                "img_mode": img_data[2],
                                "img_colour": img_data[3],
                                "id": random.random(),
                                "matched": 0,
                                "moved": 0,
                                "blocked": blocked
                            }
                        else:
                            error_count += 1
                    except KeyError:
                        continue

            nr_tot = len(json_training_data_pixels)
            logger.info(f"added {nr_tot -nr_md} records to necessary from unmatched images")
            logger.info(f"nr of errors in this part: {error_count}")
            logger.info(f"extracted a total of {nr_tot} training records")
            logger.info(f"total nr of potential pixels: {potential_pixels}")
            logger.info(f"total nr of potential images: {potential_images}")
        

            cur.close()
            db.close()
    
    except (sqlite3.OperationalError, sqlite3.IntegrityError):
        logger.error("A database error occurred:")
        logger.error(traceback.format_exc())
        return -1
    
    logger.info(f"number or training data entries in dictionary: {len(json_training_data_pixels)}")
    
    conn.close()
    
    # Construct the output filename from the input database
    root, _ = os.path.splitext(database_path)
    _, filename = os.path.split(root)

    output_path = "./training_data_output_offline_model/"
    os.makedirs(output_path, exist_ok=True)
    json_outfile = os.path.join(output_path, f"{filename}.json")

    with open(json_outfile, 'w') as fd:
        json.dump(json_training_data_pixels, fd)
    logger.info(f"Training data output to: '{json_outfile}'")
    
    return 0

if __name__ == "__main__":
    exit(main())
