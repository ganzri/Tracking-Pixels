# Copyright (C) 2022 Rita Ganz, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Script to construct different lookup tables of rankings. For each ranking it collects all the possible values (e.g. different names)
and counts how often they occur in the dataset. It then sorts the values by how often they occur, starting with the most common value
and outputs the ranking to a file.
From the url in all the http_requests for images it extracts rankings of
- the most common query parameters
- the most common domains
- the most common path pieces (i.e. a path /aaa/bbb.gif would have aaa and bbb.gif as path pieces) Note: this also creates 
path_piece-0 as the empty string, which occurs in all addresses. This is not very meaningful, so should be manually removed from the top-path-pieces file.

Usage:
    create_resources.py <db_path>

Options:
    -h --help   Displays this help message
"""

from urllib import parse
import sqlite3
import sys
import os
import re
from docopt import docopt
import logging
import traceback

logger = logging.getLogger("feature-extract")

##SQL Commands

SELECT_URL = """SELECT url FROM view_http_requests_images;"""

#SELECT_HEADER = """SELECT headers FROM view_http_requests_images;"""

SELECT_T_O = """SELECT triggering_origin FROM view_http_requests_images;"""

##
def canonical_domain(dom: str) -> str:
    """
    Transform a provided URL into a uniform domain representation for string comparison.
    from DINO: processor.py in feature extraction
    """
    canon_dom = dom.strip()
    canon_dom = re.sub("^http(s)?://", "", dom)
    canon_dom = re.sub("^www([0-9])?", "", canon_dom)
    canon_dom = re.sub("^\\.", "", canon_dom)
    canon_dom = re.sub("/$", "", canon_dom)
    return canon_dom

def setupLogger(logpath: str = "./resources/resource_extraction.log") -> None:
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


def main() -> int:
    #get urls from http_request table

    argv = None
    cargs = docopt(__doc__, argv=argv)
    
    setupLogger()

    database_path = cargs["<db_path>"]
    if not os.path.exists(database_path):
        logger.error("database path does not exist or is incorrect")
        return 1

    conn = sqlite3.connect(database_path)

    keys_in_url: Dict[str, int] = dict()
    domains_in_url: Dict[str, int] = dict()
    path_dict: Dict[str, int] = dict()
    t_o_domains: Dict[str, int] = dict()

    try:
        with conn:
            cur = conn.execute(SELECT_URL)
            url = cur.fetchall()
            for u in url:
                obj = parse.urlsplit(u[0]) # it returns a tuple, with the url in the first place
                q_dict = parse.parse_qs(obj.query)
                #remove leading www. etc and add to url set
                dom = canonical_domain(obj.netloc)
                url_path = obj.path.split("/")
                if dom in domains_in_url:
                    domains_in_url[dom] += 1
                else:
                    domains_in_url[dom] = 1
                keys = q_dict.keys()
                for k in keys:
                    if k in keys_in_url:
                        keys_in_url[k] += 1
                    else:
                        keys_in_url[k] = 1
                for p in url_path:
                    if p in path_dict:
                        path_dict[p] += 1
                    else:
                        path_dict[p] = 1
            
            cur = conn.execute(SELECT_T_O)
            origins = cur.fetchall()
            for o in origins:
                obj = parse.urlsplit(o[0])
                #remove leading www. etc and add to url set
                dom = canonical_domain(obj.netloc)
                if dom in t_o_domains:
                    t_o_domains[dom] += 1
                else:
                    t_o_domains[dom] = 1

    except (sqlite3.OperationalError, sqlite3.IntegrityError):
        return -1
    conn.close()

    outpath = "./resources/"
    os.makedirs(outpath, exist_ok=True)
    
    query_path = os.path.join(outpath, "top_query_param.csv")
    dom_path = os.path.join(outpath, "top_domains.csv")
    path_dict_path = os.path.join(outpath, "top_path_pieces.csv")
    t_o_path = os.path.join(outpath, "top_t_o_domains.csv")
    
    with open(query_path, 'w') as fd:
        for param, count in sorted(keys_in_url.items(), key=lambda item:item[1], reverse=True):
            fd.write(f"{count},{param}\n")

    with open(dom_path, 'w') as fd:
        for dom, count in sorted(domains_in_url.items(), key=lambda item:item[1], reverse=True):
            fd.write(f"{count},{dom}\n")
    
    with open(path_dict_path, 'w') as fd:
        for piece, count in sorted(path_dict.items(), key=lambda item:item[1], reverse=True):
            fd.write(f"{count},{piece}\n")
   
    with open(t_o_path, 'w') as fd:
        for domain, count in sorted(t_o_domains.items(), key=lambda item:item[1], reverse=True):
            fd.write(f"{count},{domain}\n")


    logger.info(f"nr of distinct query parameters found: {len(keys_in_url)}")
    logger.info(f"nr of distinct domains found: {len(domains_in_url)}")
    logger.info(f"nr of distinct path pieces found: {len(path_dict)}")
    logger.info(f"in {sum(domains_in_url.values())}  urls")
    logger.info(f"nr of distinct triggering origin domains found: {len(t_o_domains)}")

    return 0


if __name__ == "__main__":
    exit(main())
