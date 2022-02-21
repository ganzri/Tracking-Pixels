"""
Matches pixels fount in http_requests to their labels found in the consent notice based on
- name of pixel found in consent notice can be found in the path or netloc of the url of the http_request for an image resource
- domain of pixel found in consent notice matches netloc of url in http_request

outputs the labeled pixel data to a JSON format. Each observed pixel is an object with attributes inside the JSON. The JSON is then
used for feature extraction in the classifier.

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
            "headers": "<headers>"
      }, 
      "pixel_id_2": {
      ...
      }, 
      ...
    ]

also conducts some statistics, that could not be performed in "post_process.py" as the matching of pixels requires more functionality than SQL-queries provided.

Usage:
    create_trainingset_online.py <db_path>

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

from statistics import mean, stdev
from datetime import datetime
from typing import List, Dict, Any, Tuple
from docopt import docopt

logger = logging.getLogger("feature-extract")
time_format = "%Y-%m-%dT%H:%M:%S.%fZ"

#####################
##SQL-COMMANDS
#create taining data
MATCH_CONSENT_REQUEST = """
CREATE VIEW IF NOT EXISTS view_consent_request
    AS SELECT 
        c.id as consent_id,
        c.browser_id,
        c.visit_id,
        c.name, 
        c.domain, 
        c.cat_id,
        c.cat_name,
        c.purpose,
        c.expiry,
        c.type_name,
        c.type_id,
        h.id as req_nr_id, 
        h.extension_session_uuid,
        h.window_id,
        h.frame_id,
        h.tab_id,
        h.url,
        h.top_level_url,
        h.method,
        h.referrer,
        h.headers,
        h.request_id,
        h.triggering_origin,
        h.loading_origin,
        h.loading_href,
        h.resource_type,
        h.post_body,
        h.time_stamp,
        s.site_url
    FROM view_consent_data_images c
    JOIN site_visits s ON s.visit_id == c.visit_id
    JOIN view_first_request_images h ON c.visit_id == h.visit_id AND c.browser_id == h.browser_id 
    AND ((domain_in_url(c.domain, h.url) AND ((url_to_path(h.url) LIKE '%' || name_parser(c.name) || '%') OR (name_is_domain(c.name, c.domain)))) OR match_later(c.name, c.domain, h.request_id, h.visit_id))
    ORDER BY c.visit_id, time_stamp ASC;
"""

#some statistics
DECLARED_PIXEL_NOT_FOUND = """
CREATE VIEW IF NOT EXISTS view_declared_pixel_not_found
AS SELECT c.id, c.browser_id, c.visit_id, c.name, c.domain, c.cat_id 
FROM view_consent_data_images c 
WHERE c.id NOT IN (SELECT b.consent_id FROM view_consent_request b);
"""

MATCH_PER_CAT = """
SELECT cat_id, count() FROM view_consent_request GROUP BY cat_id ORDER BY cat_id ASC;
"""

DECLARED_PER_CAT = """
SELECT cat_id, count() FROM view_consent_data_images GROUP BY cat_id ORDER BY cat_id ASC;
"""

GET_NECESSARY_MATCH = """
SELECT * FROM view_consent_request WHERE cat_id == 0;
"""

#Query to find out which names and how often they have not been successfully matched, can be helpful to 
#check if we miss a specific pattern because of the way we do the matching or just to watch the weird
#and wonderful pixel names that exist (and fail to be matched)
NAME_NOT_MATCHED = """
SELECT name, count(*) FROM (SELECT c.id, c.browser_id, c.visit_id, c.name, c.domain 
FROM view_consent_data_images c 
WHERE c.id NOT IN (SELECT b.consent_id FROM view_consent_request b)) GROUP BY name;
"""
################END OF SQL COMMANDS

#contains all the urls connected to a single visit_id, request_id instance
urls_per_request_id = {}

def create_urls_per_request_id(conn: sqlite3.Connection) -> int:
    """
    The first time a resource is requested it gets a request_id in the http_requests table. All subsequent redirects 
    and following requests for the same resource have the same request_id. Request_id is only unique per visit.
    In rare cases the name and domain given in the consent notice matches not the first http request but one after
    a redirect. This function constructs the data structur to look up all the urls in the request-redirect chain for 
    a given visit and request id.
    """
    cur = conn.execute("Select visit_id, request_id, url FROM view_http_requests_images;")
    entries = cur.fetchall()
    global urls_per_request_id
    for e in entries:
        key = str(e[0]) + "_" + str(e[1])
        try:
            urls_per_request_id[key].add(e[2])
        except KeyError:
            urls_per_request_id[key] = {e[2]}
    return 0

##Functions needed in the SQL commands for matching pixels to consent notices
def canonical_domain(dom: str) -> str:
    """
    Transform a provided URL into a uniform domain representation for string comparison.
    from DINO: database_processing.py
    """
    canon_dom = re.sub("^http(s)?://", "", dom)
    canon_dom = re.sub("^www", "", canon_dom)
    canon_dom = re.sub("^\\.", "", canon_dom)
    return canon_dom

def name_parser(name) -> str:
    """
    We observed that some common name patterns contain # in the declaration and a number in the url. % is a wildcard symbol in 
    SQL string matching. Hence replacing # by % in the name will allow us to match these names.
    """
    return name.replace("#", "%")

def domain_in_url(domain, url) -> bool:
    c_dom = canonical_domain(domain)
    netloc = canonical_domain(parse.urlsplit(url).netloc)
    return ((netloc in c_dom) or (c_dom in netloc))

def url_to_path(url) -> str:
    return parse.urlsplit(url).path

def name_is_domain(name, domain) -> bool:
    return canonical_domain(name) == canonical_domain(domain)

def match_later(name, domain, request_id, visit_id) -> bool:
    """
    Returns true if name and domain can be matched to one of the urls in the request-redirect chain
    """
    global urls_per_request_id
    key = str(visit_id) + "_" + str(request_id)
    urls = urls_per_request_id[key]
    for u in urls:
        if domain_in_url(domain, u):
            if ((name in url_to_path(u)) or name_is_domain(name, domain)):
                #print("made it")
                return True
            else:
                continue
        else:
            continue
    return False


def setupLogger(logpath: str = "./training_data_output/extract_pixel_data.log") -> None:
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


def main()->int:
    argv = None
    cargs = docopt(__doc__, argv=argv)

    setupLogger()
    
    database_path = cargs["<db_path>"]
    if not os.path.exists(database_path):
        logger.error("Database file does not exist.")
        return 1
    
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row #enables dictionary access by column name

    logger.info("Begin training data extraction")

    json_training_data_pixels: Dict[str, Dict[str, Any]] = dict()
    
    try:
        with conn:
            #create functions needed for sql query
            conn.create_function("name_parser", 1, name_parser, deterministic=True)
            conn.create_function("domain_in_url", 2, domain_in_url, deterministic=True)
            conn.create_function("url_to_path", 1, url_to_path, deterministic = True)
            conn.create_function("name_is_domain", 2, name_is_domain, deterministic = True)
            conn.create_function("match_later", 4, match_later, deterministic=True)
            create_urls_per_request_id(conn)

            conn.execute(MATCH_CONSENT_REQUEST)
            cur = conn.execute("SELECT count() FROM view_consent_request;")
            logger.info(f"number of pixels that could be matched to a declaration {cur.fetchone()[0]}")
            
            cur = conn.execute(DECLARED_PIXEL_NOT_FOUND)
            cur = conn.execute("SELECT count() FROM view_declared_pixel_not_found;")
            logger.info(f"number of pixels declared in a consent notice but not found {cur.fetchone()[0]}")
            cur = conn.execute("SELECT count() FROM view_consent_data_images")
            logger.info(f"total number of pixels declared in consent notices, all categories {cur.fetchone()[0]}")
            
            crcounts: List[int]= []
            for i in range(-1, 6):
                cur.execute(f'SELECT COUNT(visit_id) AS count FROM (SELECT DISTINCT visit_id, request_id FROM view_consent_request WHERE cat_id == {i});')
                crcounts.append(int(cur.fetchone()["count"]))

            logger.info("Nr of matches per categorie\n")
            logger.info(f"Unknown Count:       {crcounts[0]}\n")
            logger.info(f"Necessary Count:     {crcounts[1]}\n")
            logger.info(f"Functional Count:    {crcounts[2]}\n")
            logger.info(f"Analytical Count:    {crcounts[3]}\n")
            logger.info(f"Advertising Count:   {crcounts[4]}\n")
            logger.info(f"Unclassified Count:  {crcounts[5]}\n")
            logger.info(f"Social Media Count:  {crcounts[6]}\n\n")

            # noise analysis
            logger.info("starting noise analysis")
            distinct_pixels = set()
            majority_class: Dict[str, [5]] = dict()
            nr_declared_pixels = 0

            cur = conn.execute("SELECT * from view_consent_request;")
            declaration = cur.fetchall()
            nr_declared_pixels =len(declaration)
            logger.info(f"total nr of matches: d = {len(declaration)}")
            for line in declaration:
                name = line[3]
                domain = line[4]
                purpose = line[5]
                key = name+domain
                distinct_pixels.add(key)

                if key in majority_class:
                    #print(majority_class[key])
                    majority_class[key][purpose] +=1
                else:
                    majority_class[key] = [0, 0, 0, 0, 0]
                    majority_class[key][purpose] = 1

            logger.info(f"distinct pixel declarations matched: {len(distinct_pixels)}")
            distinct_third_party = 0
            nr_occurence_third_party = 0
            deviation_from_majority = 0

            for key in majority_class:
                if sum(majority_class[key]) > 1:
                    distinct_third_party +=1
                    nr_occurence_third_party += sum(majority_class[key])
                    s = sort(majority_class[key])
                    deviation_from_majority += sum(s[:4])

            #how many unique pixels that occur more than once
            logger.info(f"unique third party pixels matched: {distinct_third_party}")
            logger.info(f"declared third party pixels matched (can match the same pixel multiple times): a = {nr_occurence_third_party}")
            logger.info(f"nr of matches deviating from majority class: b = {deviation_from_majority}")
            logger.info(f"lower bound for noise in third party pixels: b / a ) = {deviation_from_majority/nr_occurence_third_party}")
            logger.info(f"lower bound for noise all pixels: b / d = {deviation_from_majority/nr_declared_pixels}")
            
            #create training data json
            cur = conn.cursor()
            cur.execute("Select * FROM view_consent_request;")
            for row in cur:
                if row["expiry"].startswith("+0"):
                    continue #ignore rare cases where expire is year 10'000 and upwards
                    
                cat_id = int(row["cat_id"])
                if not (0 <= cat_id <= 3):
                    continue #ignore anything not in our core categories

                json_pixel_key = str(row["visit_id"]) + "_" + str(row["request_id"])
                
                json_training_data_pixels[json_pixel_key] = {
                    "visit_id": row["visit_id"],
                    "request_id": row["request_id"],
                    "name": row["name"],
                    "url": row["url"],
                    "first_party_domain": row["site_url"],
                    "label": cat_id,
                    "triggering_origin": row["triggering_origin"],
                    "headers": row["headers"]
                }
                #process variable data (or headers) and append
                #json_training_data_pixels[json_pixel_key].append({})

            cur.close()
    
    except (sqlite3.OperationalError, sqlite3.IntegrityError):
        logger.error("A database error occurred:")
        logger.error(traceback.format_exc())
        return -1
    
    logger.info(f"number or training data entries in dictionary: {len(json_training_data_pixels)}")
    
    conn.close()

    # Construct the output filename from the input database
    root, _ = os.path.splitext(database_path)
    _, filename = os.path.split(root)

    output_path = "./training_data_output/"
    os.makedirs(output_path, exist_ok=True)
    json_outfile = os.path.join(output_path, f"{filename}.json")

    with open(json_outfile, 'w') as fd:
        json.dump(json_training_data_pixels, fd)
    logger.info(f"Training data output to: '{json_outfile}'")
    
    return 0

if __name__ == "__main__":
    exit(main())
