This folder contains scripts to create the trainig data set for the classifier plus some scripts to analyse the data, 
compare it to filterlists etc.

Scripts:
First step: create resources and json training sample file from SQL tables
* [create_resources.py] used to create the resources from the SQL tables (ranking for top netlocs, path pieces etc)
* [image_features.py] to create the resource tables for image format and mode (though it just prints them, one has to manually add them to the resource folder)
* [create_trainingset_online.py] first step of extracting the training samples from the SQL database, outputs a json to be used by its offline equivalent (see next)
* [create_trainingset_offline_model.py] takes the output of above, and adds the information from the HTTP response and LevelDB (both take a long time, that is why they are separat)

Second step: analyse outcome of creating training set, reclassify, see folder training_data_output_offline

Third step: create sparse matrix representation of the training samples to be used by the classifier
* [training_validation_split.py] split the json into training and validation set. This is only needed for the next steps, if one only wants to train the model this is not needed. 
* [prepare_training_data.py] creates the sparse matrix representation of the samples. Uses scripts in the folder feature_extraction, and the resources.

Fourth step: train classifier, see folder classifier

Fifth step: analyse the outcome.
* [analyse.py] prints samples of interest as stored in a json to the console together with the decisions by filter lists (which list would block it based on what rule)
* [analyse_gt_aa_nb.py] similar to analyse, but removes overly common samples, used for the bigger datasets to analyse
* [figure_mismatch.py] plots figure 5.3 of thesis.
* [in_necessary.py] calculates how many training samples from a given piece of url are found in the necessary category.
* [domain_to_query.py] This script returns an overview of netlocs that use either a given path piece or a query parameter.

Folders:
* [abp_blocklist_parser] code from (https://github.com/englehardt/abp-blocklist-parser) Copyright © 2018 Steven Englehardt and other contributors; This is used to query the filter lists, whether they would block a URL. This folder also contains whole_list.txt the EasyList as of 27.01.2022 and easyprivacy.txt the EasyPrivacy list as of 28.01.2022, Credits  "The EasyList authors (https://easylist.to/)", dual licensed under the GNU General Public License version 3 of the License, and Creative Commons Attribution-ShareAlike 3.0 Unported.
* [classifier] contains code to train the classifier, some stats
* [feature_extraction] contains the pixel_processor used by prepare_training_data and a json defining the features, plus utils. Scripts in this folder plus prepare_training_data plus the lookup tables in the folder resources are needed for the creation of sparse matrices to be used by the classifier.
* [processed_features] will contain the output of prepare_training_data
* [resources] txt files of the different rankings of features, created by create_resources.py, used in the feature extraction
* [training_data_output] output of create_trainingset_online.py
* [training_data_output_offline_model] output of create_trainingset_offline_model.py and scripts to reassing categories and some stats.

