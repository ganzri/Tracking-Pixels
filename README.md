# Scripts used for Bachelor Thesis "Understanding GDPR compliance of tracking pixel declarations using privacy filter lists"

* [Description](#description)
* [Repository Contents](#repository-contents)
* [Credits](#credits)
* [License](#license)

## Description
This repository contains code developed as part of a Bachelor Thesis. We collected tracking pixels and images together with their purpose labels from websites with detailed consent notices. Using this dataset we trained an XGBoost model to predict purposes, and compared classifications by the model, the consent declarations, and the filter lists.

The thesis can be found here:
https://www.research-collection.ethz.ch/handle/20.500.11850/535362

For the data collection the webcrawler from Dino Bollinger was used, which can be found here:
(https://github.com/dibollinger/CookieBlock-Consent-Crawler)

This repository contains the scripts used to prepare the data for the classifier, train the classifier and conduct aditional analysis.

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
