analyse outcome of creating training set, reclassify
* [cat_stats.py] can be used to get some stats on how many training samples per category the training sample json contains.
* [analyse_functional.py] prints all functional samles to the console to be analysed manually, except for facebook and google.com, as those are definitely not functional.
* [header_fields.py] Calculates how often each header field occurs in the dataset and which header fields occur. It also produces some plots on colour, entropy, transparency distribution in the different classes.
* [large_pixels.py] To answer (partially) the question whether there are large pixels (larger than 1x1).
* [reclassify.py] Moves pixels from 9 very common netlocs (e.g. Facebook, Google Analytics) from necessary and functional to the correct class
* [noise_analysis.py] Script to estimate the noise in the dataset based on deviation from majority class. Uses original and reclassified data
