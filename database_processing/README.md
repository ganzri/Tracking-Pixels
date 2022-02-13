# Database Scripts

This subfolder contains scripts to perform operations on the database produced by the consent label crawler.

This includes altering/sanetizing the database for later convenience using SQL commands as well as extracting some statistics

## Folder Contents

`sql_commands/`: Contains queries, commands and assertions which may be useful. Some of these are used by `post_process_db.py`.

`stats/`: Target folder for the statistics files as output by: `post_process_db.py`

`post_process_db.py`: 

   First, backs up the database. Then, cleans empty tables out of the database and runs some sanity checks on them.
   Next, it creates a number of views inside the database to allow easier analysis of certain datapoints in the database.
   
   Finally, it runs a number of queries to output a large range of statistics. This includes:
   - Total URLs visited.
   - Number of successful CMP crawls.
   - Number of CMP crawl failures.
   - How many crawls were interrupted before retrieving the CMP data?
   - How many crawls were interrupted during the browse phase?
   - All errors ordered by type.
   - Number of http requests and responses for unique images in database
   - Number of declarations in database (for pixels and cookies)
   - pixel counts separated by purpose.

## License

