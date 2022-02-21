# Domain Sources
This folder contains domain sources that were used with the presence crawler.

The purpose of the scripts in the folder is to collect this data, and to filter duplicates. Split_data and combine_sets.py can be used to split the data and recombine, if the crawl should not be run on all the data at once.

## Folder Contents
* `Tranco_Lists/`: Contains the Tranco domain ranking as of Dez 2021 (and subsets of the lists)
	* `tranco_PJWJ_world.csv` Tranco top 1 million worldwide domains generated on 03 December 2021, available at https://tranco-list.eu/list/PJWJ.
	* `tranco_Q574_europe.csv` Tranco top european domains generated on 04 December 2021, available at https://tranco-list.eu/list/Q574.
* `generate_diff_set.py`: Used to compare the Tranco lists and produce a list of domains without overlap.
* `split_data.py`: can be used to split the Tranco lists into chunks and run the crawl on a subset.
