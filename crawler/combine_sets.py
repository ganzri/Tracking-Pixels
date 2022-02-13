"""
Script to recombine the domains returned by the presence crawl, if that was run on split data
Creates the domain lists that can then be used by the consent crawler
If the presence crawl was run on all the data at once, this is not necessary
"""

worldwide = ["filtered_domains_set_a", "filtered_domains_set_b", "filtered_domains_set_c"]
europe = ["filtered_domains_europe_a","filtered_domains_europe_b", "filtered_domains_europe_c"]
files = ["bot_responses.txt", "cookiebot_responses.txt", "crawler_timeouts.txt", "failed_urls.txt", "http_responses.txt", "nocmp_responses.txt", "onetrust_responses.txt", "termly_responses.txt"]
all_regions=[worldwide, europe]
all_folders = worldwide + europe

for folder in all_regions: #do work for both european and worldwide datasets
    for f in files: # for each type of file, ie. cookiebot_responses
        if folder == worldwide:
            outpath = "./filtered_domains_all/worldwide/" + f #e.g. ./filtered_domains_all/europe/cookiebot_responses.txt
        else:
            outpath = "./filtered_domains_all/europe/" + f
        count = 0
        with open(outpath, 'w') as fd:
            #get data for the same file from each folder and combine
            for s in folder:
                inpath = "./" + s + "/" + f
                with open(inpath, 'r') as inf:
                    for line in inf:
                        fd.write(line)
                        count +=1
            print(f"file {f} with {count} entries writen to {outpath}")

for f in files: # for each type of file, ie. cookiebot_responses
    outpath = "./filtered_domains_all/" + f #e.g. ./filtered_domains_all/cookiebot_responses.txt
    count = 0
    with open(outpath, 'w') as fd:
        #get data for the same file from each folder and combine
        for s in all_folders:
            inpath = "./" + s + "/" + f
            with open(inpath, 'r') as inf:
                for line in inf:
                    fd.write(line)
                    count +=1
        print(f"file {f} with {count} entries writen to {outpath}")

