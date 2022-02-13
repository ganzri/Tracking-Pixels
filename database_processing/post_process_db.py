#!/bin/python3
# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
# modified RG
"""
This script runs a series of SQL commands on the specified database to remove empty tables
and add Views in order to have a better overview of the data contained within.

Also extracts some useful statistics to analyze the information in the database.

Usage:
    post_process_db.py <db_path>
"""

import sqlite3
import logging
import traceback
import os

from typing import List
from docopt import docopt

logger = logging.getLogger("main")

assertion_path = "sql_commands/assertions/"
command_path = "sql_commands/commands/"

def read_sql_script(path):
    """ utility to read sql script file """
    with open(path, 'r') as fd:
        return fd.read()


def sanity_check(conn: sqlite3.Connection) -> None:
    """
    Executes the scripts stored at the assertion path in order to verify basic
    sanity properties on the database. If these are violated, something probably
    went horribly wrong during the crawl.

    If this is the case, it will report a warning in the log file.
    @param conn: Active SQLite3 database connection
    """
    assert_files = [os.path.join(assertion_path, f) for f in os.listdir(assertion_path) if f.endswith(".sql")]
    logger.info("Executing sanity checks...")
    for a in assert_files:
        try:
            script = read_sql_script(a)
            with conn:
                cur = conn.execute(script)

                # expected format: "True" / "False"
                passed = bool(cur.fetchone()[0])
                if not passed:
                    logger.warning(f"Sanity check failed: {a}")
                cur.close()
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            logger.error(f"Failed to execute script at {a}")
            logger.error(traceback.format_exc())


def create_backup(dbpath: str, conn: sqlite3.Connection) -> int:
    """
    Creates a backup of the specified database, using the specified path dbpath,
    and an open database connection from which the backup is created.

    Backup will have the filepath "<dbpath>_backup". Will not overwrite existing backup.
    @param dbpath: Path + Filename of the original database
    @param conn: Active connection to said original database
    @return: 0 if a backup was created, 1 if it failed.
    """
    try:
        def progress(status, remaining, total):
            logger.debug(f"Copy status: {status}")
            logger.debug(f'Copied {total - remaining} of {total} pages...')

        backup_path = dbpath + '_backup'
        if os.path.exists(backup_path):
            logger.info(f"Previous backup already exists, skipping...")
            return 0

        logger.info(f"Creating a database backup at {backup_path}")
        bck = sqlite3.connect(backup_path)
        with bck:
            conn.backup(bck, pages=1, progress=progress)
        bck.close()
        return 0
    except (sqlite3.IntegrityError, sqlite3.OperationalError):
        logger.error(f"Failed to create backup of {dbpath}")
        logger.error(traceback.format_exc())
        return 1


def execute_db_transformation_commands(conn: sqlite3.Connection) -> None:
    """
    Executed commands intended to transform the database.
    Commands include: Removing leftover tables, setting up useful views.
    @param conn: Active SQLite3 database connection
    """
    command_files = [os.path.join(command_path, f) for f in os.listdir(command_path) if f.endswith(".sql")]
    logger.info("Executing database setup...")
    for c in sorted(command_files):
        script = read_sql_script(c)
        try:
            with conn:
                cur = conn.executescript(script)
                cur.close()
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            logger.error(f"Failed to execute script at {c}")
            logger.error(traceback.format_exc())
        else:
            logger.info(f"Successfully executed script {c}")


def extract_debug_statistics(conn: sqlite3.Connection, debug_stats_path="./debug_stats.txt") -> None:
    """
    Runs a series of SQL queries to extract debug statistics from the consent crawl database.
    Examples: Number of successful crawls, interrupted crawls, CMP errors etc.

    These statistics will be stored in the specified stats file path.
    @param conn: Active sqlite3 database connection.
    @param debug_stats_path: Path to store the statistics in.
    """
    try:
        with conn:
            cur = conn.cursor()

            # Get total urls
            cur.execute("SELECT COUNT(visit_id) AS count FROM site_visits;")
            total_urls: int = int(cur.fetchone()["count"])

            # successful CMP crawls
            cur.execute("SELECT COUNT(visit_id) AS count FROM consent_crawl_results WHERE crawl_state == 0;")
            successful_cmp_count: int = int(cur.fetchone()["count"])

            # consent crawls with non-zero status code
            cur.execute("SELECT COUNT(visit_id) AS count FROM view_failed_consent_crawls;")
            failed_cmp_count: int = int(cur.fetchone()["count"])

            cur.execute("SELECT COUNT(visit_id) AS count FROM view_failed_visits_consent;")
            interrupts_during_consent_crawl: int = int(cur.fetchone()["count"])

            cur.execute("SELECT COUNT(visit_id) AS count FROM view_failed_visits_browse;")
            interrupts_during_browse: int = int(cur.fetchone()["count"])

            cur.execute('SELECT COUNT(visit_id) AS count FROM view_consent_crawl_results WHERE crawl_state == 0 AND browse_interrupted == "False";')
            successful_crawls_overall: int = int(cur.fetchone()["count"])

            interrupts_total: int = interrupts_during_browse + interrupts_during_consent_crawl

            interrupt_error_reports: List = []
            cur.execute('SELECT visit_id, site_url, error_type, error_report FROM view_visits_with_errors WHERE interrupted == "True" ORDER BY error_type;')
            for line in cur:
                report = f'{line["visit_id"]} -- {line["site_url"]} -- Type: {line["error_type"]}'
                if line["error_report"]:
                    report += f' -- Details: {str(line["error_report"]).strip()}\n'
                else:
                    report += "\n"
                interrupt_error_reports.append(report)

            cmp_error_reports: List = []
            cur.execute('SELECT * FROM view_failed_consent_crawls ORDER BY crawl_state ASC;')
            ctype = -1
            for line in cur:
                if ctype != int(line["crawl_state"]):
                    ctype = int(line["crawl_state"])
                    cmp_error_reports.append(f"\nType {ctype}:\n")
                report = f'{line["visit_id"]} -- {line["site_url"]} -- Type: {line["crawl_state"]} -- Details: {str(line["report"]).strip()}\n'
                cmp_error_reports.append(report)

            cur.close()

            with open(debug_stats_path, 'w') as fd:
                fd.write("\n## Debug Statistics\n")
                fd.write(f"URL Total: {total_urls}\n")
                fd.write(f"  -- success: {successful_crawls_overall}\n")
                fd.write(f"  -- failed:  {total_urls - successful_crawls_overall}\n")
                fd.write(f"Crawls Interrupted: {interrupts_total}\n")
                fd.write(f"  -- in consent phase: {interrupts_during_consent_crawl}\n")
                fd.write(f"  -- in browse phase:  {interrupts_during_browse}\n")
                # fd.write(f"Crawls Uninterrupted: {total_urls - interrupts_total }\n")
                fd.write("Consent Management Platform Data:\n")
                fd.write(f"  -- found:     {successful_cmp_count}\n")
                fd.write(f"  -- not found: {failed_cmp_count}\n")

                fd.write("\n## Interrupt Errors:\n")
                for line in interrupt_error_reports:
                    fd.write(line)

                fd.write("\n## Consent Crawl Errors:\n")
                for line in cmp_error_reports:
                    fd.write(line)

    except (sqlite3.IntegrityError, sqlite3.OperationalError):
        logger.error("A database error occurred:")
        logger.error(traceback.format_exc())
    else:
        logger.info("Debug statistics successfully extracted.")


def extract_content_statistics(conn: sqlite3.Connection, stats_path: str) -> None:
    """
    Runs a series of SQL queries to extract content statistics from the consent crawl database.
    Examples: number of records collected per category, training data records, etc.

    These statistics will be stored in the specified stats file path.
    @param conn: Active sqlite3 database connection.
    @param stats_path: Path to store the statistics in.
    """
    try:
        with conn:
            cur = conn.cursor()


            cur.execute('SELECT COUNT("True") AS count FROM consent_data;')
            total_consent_records : int = int(cur.fetchone()["count"])

            cur.execute('SELECT COUNT("True") AS count FROM view_consent_data_images;')
            pixel_consent_records : int = int(cur.fetchone()["count"])

            crcounts: List[int]= []
            for i in range(-1, 6):
                cur.execute(f'SELECT COUNT(visit_id) AS count FROM (SELECT DISTINCT visit_id, name, domain, cat_id, cat_name, purpose, expiry, type_name, type_id FROM consent_data WHERE cat_id == {i});')
                crcounts.append(int(cur.fetchone()["count"]))

            #count number of unique pixels (unique meaning distinct (name, domain) pair from 
            #consent declaration)
            cur.execute("SELECT COUNT() AS count FROM view_unique_consent_pixels;")
            unique_pix_declarations = int(cur.fetchone()["count"])
            

            #count nr of pixels declared in the consent declaration per categorie
            pxcounts: List[int] = []
            for i in range(-1, 6):
                cur.execute(f'SELECT COUNT("True") AS count FROM (SELECT DISTINCT visit_id, name, domain, cat_id, cat_name, purpose, expiry, type_name, type_id FROM view_consent_data_images) as img WHERE img.cat_id == {i};')
                pxcounts.append(int(cur.fetchone()["count"]))

            #count nr of unique pixels declared per categorie (could be the case that a pair occurs in more             #than one categorie and hence if we add this up it might be larger than total unique count
            u_pxcounts: List[int] = []
            for i in range(-1, 6):
                cur.execute(f'SELECT COUNT("True") AS count FROM (SELECT DISTINCT name, domain FROM view_consent_data_images as im WHERE im.cat_id == {i});')
                u_pxcounts.append(int(cur.fetchone()["count"]))
            
            cur.execute('SELECT COUNT("True") AS count FROM view_first_request_images;')
            img_requests = int(cur.fetchone()["count"])
            
            cur.execute('SELECT COUNT("True") AS count FROM http_responses WHERE content_hash <> "NULL";')
            img_responses = int(cur.fetchone()["count"])

            cur.close()

            with open(stats_path, 'w') as fd:
                fd.write("\n## Content Statistics\n")
                fd.write(f"Total CMP Records:       {total_consent_records}\n")
                fd.write(f"Total CMP Records for Pixels only: {pixel_consent_records}\n")
                fd.write(f"Unique Pixels in CMPs:          {unique_pix_declarations}\n")
                fd.write(f"Nr of different image resources requested in http_requests:  {img_requests}\n")
                fd.write(f"Nr of different image resources delivered in http_responses: {img_responses}\n")
                fd.write("\n")

                fd.write("## Count for each category of data collected from the Consent Management Platforms\n")
                fd.write(f"Unknown Count:       {crcounts[0]}\n")
                fd.write(f"Necessary Count:     {crcounts[1]}\n")
                fd.write(f"Functional Count:    {crcounts[2]}\n")
                fd.write(f"Analytical Count:    {crcounts[3]}\n")
                fd.write(f"Advertising Count:   {crcounts[4]}\n")
                fd.write(f"Unclassified Count:  {crcounts[5]}\n")
                fd.write(f"Social Media Count:  {crcounts[6]}\n\n")

                fd.write("## Count for each category of data collected from the Consent Management Platforms for Pixels only\n")
                fd.write(f"Unknown Count:       {pxcounts[0]}\n")
                fd.write(f"Necessary Count:     {pxcounts[1]}\n")
                fd.write(f"Functional Count:    {pxcounts[2]}\n")
                fd.write(f"Analytical Count:    {pxcounts[3]}\n")
                fd.write(f"Advertising Count:   {pxcounts[4]}\n")
                fd.write(f"Unclassified Count:  {pxcounts[5]}\n")
                fd.write(f"Social Media Count:  {pxcounts[6]}\n\n")

                fd.write("## Count for each category of data collected from the Consent Management Platforms for unique Pixels only\n")
                fd.write(f"Unknown Count:       {u_pxcounts[0]}\n")
                fd.write(f"Necessary Count:     {u_pxcounts[1]}\n")
                fd.write(f"Functional Count:    {u_pxcounts[2]}\n")
                fd.write(f"Analytical Count:    {u_pxcounts[3]}\n")
                fd.write(f"Advertising Count:   {u_pxcounts[4]}\n")
                fd.write(f"Unclassified Count:  {u_pxcounts[5]}\n")
                fd.write(f"Social Media Count:  {u_pxcounts[6]}\n\n")
                

    except (sqlite3.IntegrityError, sqlite3.OperationalError):
        logger.error("A database error occurred:")
        logger.error(traceback.format_exc())
    else:
        logger.info("Content Statistics successfully extracted.")


def setupLogger(logdir:str, loglevel:str) -> None:
    """
    Set up the logger instance, which will write its output to a log file.
    :param logdir: Directory for the log file.
    :param loglevel: Log level at which to record.
    """
    loglevel = logging.getLevelName(loglevel)
    logger.setLevel(loglevel)

    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, "db_transform.log")

    with open(logfile, 'w') as fd:
        pass

    """ Enables logging to stderr """
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt="%Y-%m-%d-%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # log file output
    fh = logging.FileHandler(filename=logfile, mode="w", encoding="utf8")
    fh.setLevel(loglevel)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def main() -> int:
    """
    Perform sanity check, backup, transform db, extract debug stats, extract content stats.
    @return: exit code, 0 for success
    """
    argv = None
    # argv = ["./example_db/example_crawl_20210213_153228.sqlite"]

    args = docopt(__doc__, argv=argv)

    setupLogger(".", "INFO")

    database_path = args["<db_path>"]
    if not os.path.exists(database_path):
        logger.error("Database does not exist.")
        return 1

    root, _ = os.path.splitext(database_path)
    _, filename = os.path.split(root)
    debug_statistics_path = f"stats/debug_stats_{filename}.txt"
    content_statistics_path = f"stats/content_stats_{filename}.txt"

    # enable access by column name
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row

    sanity_check(conn)
    status = create_backup(database_path, conn)
    if status != 0:
        logger.error("Backup failed, aborting further processing.")
        return status
    execute_db_transformation_commands(conn)

    extract_debug_statistics(conn, debug_statistics_path)
    extract_content_statistics(conn, content_statistics_path)

    conn.close()

    return 0


if __name__ == "__main__":
    exit(main())

