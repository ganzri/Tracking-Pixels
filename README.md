# Scripts used for Bachelor Thesis "Understanding GDPR compliance of tracking pixel declarations using privacy filter lists"

* [Description](#description)
* [Repository Contents](#repository-contents)
* [Credits](#credits)
* [License](#license)

## Description


## Repository Contents
The order in which the subfolders are presented follows the order in which they would be used.
Please note that the repositories all contain a README explaining in more detail, what they do.
* [domain_sources] Contains the domain sources that were used with the presence crawler. The purpose of the scripts in the folder is to filter duplicates.
 * [filtered_domains_all] Contains the results of the presence crawl. Subfolder europe contains the output of running the presence crawl on the european domains, subfolder worldwide on all the domains. 
* [database_processing] Contains scripts and SQL commands to clean up the data from the consent crawl, create useful views and extract some statistics
 * [create_trainingset_and_classifier] contains the scripts to create the trainingset for the classifer and the classifier itself.
 
__Thesis supervision:__
* Karel Kubicek
* Prof. Dr. David Basin
* Information Security Group at ETH Zürich
---

Includes code from:
* [BlockListParser] (https://github.com/englehardt/abp-blocklist-parser) Copyright © 2018 Steven Englehardt and other contributors

## License

__Copyright © 2022, Rita Ganz, Department of Computer Science at ETH Zürich, Information Security Group__
