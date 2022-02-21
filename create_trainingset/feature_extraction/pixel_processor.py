# heavily relying on Dino's cookie processor
# inputs: json file with pixel training data, json file with feature definition
"""
Input JSON structure:
    [
     "pixel_id_1": {
            "visit_id": "<visit_id>",
            "request_id": "<request_id>",
            "name": "<name>",
            "url": "<url>",
            "first_party_domain": "<site_url>",
            "label": [0-3],
            "triggering_origin": "<triggering_origin>",
            "headers": "<headers>",
            "img_format": img_data[0],
            "img_size": "(width, height)",
            "img_mode": img_data[2],
            "img_colour": "(r,g,b,alpha)"
            "matched": "<matched>",
            "moved": "<moved>",
            "blocked": "(EasyPrivacy, EasyList)"

      },
      "pixel_id_2": {
      ...
      },
      ...
    ]

"""
# imports essential
import base64
import csv
import json

import re
from statistics import mean, stdev
#import urllib.parse
import zlib
from collections import Counter
from math import log

import scipy.sparse
from sklearn.datasets import dump_svmlight_file
import xgboost as xgb
import random
import difflib
#from Levenshtein import distance as lev_distance

from .utils import (url_parse_to_uniform, load_lookup_from_csv, url_to_uniform_domain, split_delimiter_separated,
                   check_flag_changed, try_decode_base64, try_split_json, delim_sep_check)

# Non-essential
import logging
import time
import pickle

from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Set
from urllib import parse

logger = logging.getLogger("feature-extract")

class PixelFeatureProcessor:
    def __init__(self, feature_def: str) -> None:
        """takes as input features defined in 'features.json'"""

        #read feature definition from json file
        with open(feature_def) as fmmap:
            self.feature_mapping: Dict[str, Any] = json.load(fmmap)

        #find correct delimiter in csv files
        self.csv_sniffer: csv.Sniffer = csv.Sniffer()

        #compute expected number of features based on feature mapping
        self.num_pixel_features: int = 0
        funcs = 0
        f_enabled = 0
        for f in self.feature_mapping["per_pixel_features"]:
            funcs += 1
            if f["enabled"]:
                f_enabled +=1
                self.num_pixel_features += f["vector_size"]
        logger.info(f"Number of per-pixel functions: {funcs}")
        logger.info(f"Number of per-pixel functions enabled: {f_enabled}")
        logger.info(f"Number of per-pixel features: {self.num_pixel_features}")

        self.num_features: int = self.num_pixel_features
        #TODO update above it there are other than per pixel features (I think there should not be)

        # tracks the current features in sparse representation
        self._row_indices: List[int] = list()
        self._col_indices: List[int] = list()
        self._data_entries: List[float] = list()
        self._labels: List[int] = list()

        # cursor for sparse features
        self._current_row: int = 0
        self._current_col: int = 0    
        
        #TODO add other lookup tables for new features as required
        # Lookup table: Domain -> Rank
        self._top_domain_lookup: Optional[Dict[str, int]] = None
        self._top_query_param_lookup: Optional[Dict[str, int]] = None
        self._top_path_piece_lookup: Optional[Dict[str, int]] = None
        self._top_t_o_domain_lookup: Optional[Dict[str, int]] = None
        self._format_lookup: Optional[Dict[str, int]] = None
        self._mode_lookup: Optional[Dict[str, int]] = None

        # This set is required to limit false positives. These are all the separators recognized as valid
        self.valid_csv_delimiters: str = ",|#:;&_.-"

        # Strings that identify boolean values.
        self.truth_values: re.Pattern = re.compile(r"\b(true|false|yes|no|0|1|on|off)\b", re.IGNORECASE)
        
        # Setup external resources through CSV and TXT file data
        logger.debug("Setting up lookup data...")
        
        for feature in self.feature_mapping["per_pixel_features"]:
            if feature["enabled"] and "setup" in feature:
                assert hasattr(self, feature["setup"]), f"Setup function not found: {feature['setup']}"
                logger.debug(f"Running setup method: {feature['setup']}")
                function = getattr(self, feature["setup"])
                function(source=feature["source"], vector_size=feature["vector_size"], **feature["args"])

        logger.debug("Lookup setup complete.")

    #
    ## Internal Data Handling
    ## Methods used to construct the sparse matrix representation
    #

    def _reset_col(self) -> None:
        """ Reset column position and verify feature vector size. """
        assert self.num_features == self._current_col, f"Inconsistent Feature Count {self.num_features} and {self._current_col}"
        self._current_col = 0

    def _increment_row(self, amount: int = 1) -> int:
        """ Each row of the matrix stores features for a single pixel instance (including all updates).
            :param amount: By how much to shift the cursor
        """
        self._current_row += amount
        return self._current_row

    def _increment_col(self, amount: int = 1) -> int:
        """ Increment the internal column counter, i.e. change feature index.
            :param amount: By how much to shift the cursor
        """
        self._current_col += amount
        return self._current_col

    def _insert_label(self, label: int) -> None:
        """ Append label to the internal listing.
        :param label: Label to append, as integer.
        """
        self._labels.append(label)

    def _multi_insert_sparse_entries(self, data: List[float], col_offset: int = 0) -> None:
        """
        Insert multiple sparse entries -- required in certain cases
        :param data: Floating point entries to insert into the sparse representation.
        :param col_offset: By how many entries to offset the insertion from the current cursor.
        """
        c = 0
        for d in data:
            self._row_indices.append(self._current_row)
            self._col_indices.append(self._current_col + col_offset + c)
            self._data_entries.append(d)
            c += 1

    def _insert_sparse_entry(self, data: float, col_offset: int = 0) -> None:
        """
            Updates sparse representation arrays with the provided data.
            :param data: Data entry to insert into the sparse matrix.
            :param col_offset: Used when position of one-hot vector is shifted from current cursor.
        """
        self._row_indices.append(self._current_row)
        self._col_indices.append(self._current_col + col_offset)
        self._data_entries.append(data)
    ##
    ## Outwards-facing methods:
    ##

    def reset_processor(self) -> None:
        """ Reset all data storage -- to be used once a matrix is fully constructed, and another needs to be generated. """
        self._row_indices.clear()
        self._col_indices.clear()
        self._data_entries.clear()
        self._labels.clear()
        self._current_col = 0
        self._current_row = 0

    def retrieve_labels(self) -> List[int]:
        """ Get a copy of the current label list. """
        return self._labels.copy()


    def retrieve_label_weights(self, num_labels: int) -> List[float]:
        """
        Compute weights from the label array in order to counter class imbalance.
        Assumption: Labels start from 0, up to num_labels.
        RG added a small 'error' to avoid division by zero, which is likely for pixels
        :param num_labels: Maximum label index. Final index ranges from 0 to num_labels.
        :return: Inverse frequency of each label.
        """
        num_total = len(self._labels)
        inverse_ratio = [num_total / (self._labels.count(i) + 1) for i in range(num_labels)]
        logger.info(f"Computed Weights: {inverse_ratio}")
        return [inverse_ratio[lab] for lab in self._labels]


    def retrieve_feature_names_as_list(self) -> List[str]:
        """returns list of feature names in a sequential list"""
        feat_list = []
        feat_cnt = 0
        for feature in self.feature_mapping["per_pixel_features"]:
            if feature["enabled"]:
                for i in range(feature["vector_size"]):
                    feat_list.append(str(feat_cnt + i) + " " + feature["name"] + f"-{i} i")
                feat_cnt += feature["vector_size"]
        logger.info(f"feature name length: {len(feat_list)}")
        return feat_list

    def retrieve_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """ From the collected data, construct a CSR format sparse matrix using scipy. """
        assert len(self._data_entries) > 0, "No features stored by processor!"
        return scipy.sparse.csr_matrix((self._data_entries, (self._row_indices, self._col_indices)))

    def retrieve_xgb_matrix(self, include_labels: bool, include_weights: bool) -> xgb.DMatrix:
        """
        From the collected data, construct a xgb binary-format matrix.
        :param include_labels: If true, will include labels inside the binary.
        :param include_weights: If true, will include weights for each label inside the binary.
        :return: XGB DMatrix
        """
        assert len(self._data_entries) > 0, "No features stored by processor!"
        assert (not include_labels and not include_weights) or len(self._labels) > 0, "No labels stored by processor!"
        sparse_mat: scipy.sparse.csr_matrix = self.retrieve_sparse_matrix()
        logger.info(f"dimension of data: {sparse_mat.shape}")
        labels: Optional[List[int]] = self.retrieve_labels() if include_labels else None
        weights: Optional[List[float]] = self.retrieve_label_weights(num_labels=4) if include_weights else None
        return xgb.DMatrix(sparse_mat, label=labels, weight=weights, feature_names=self.retrieve_feature_names_as_list())

    def dump_sparse_matrix(self, out_path: str, dump_weights: bool = True) -> None:
        """
        Dump the sparse matrix of features extracted from the cookies.
        :param out_path: filename for the pickled sparse matrix
        :param dump_weights: if true, will also dump the instance weights
        """
        dtrain = self.retrieve_sparse_matrix()
        with open(out_path, 'wb') as fd:
            pickle.dump(dtrain, fd)

        feature_names = self.retrieve_feature_names_as_list()
        with open(out_path + ".feature_names", 'wb') as fd:
            pickle.dump(feature_names, fd)

        labels = self.retrieve_labels()
        with open(out_path + ".labels", 'wb') as fd:
            pickle.dump(labels, fd)

        if dump_weights and len(Counter(labels).keys()) == 4:
            weights = self.retrieve_label_weights(num_labels=4)
            with open(out_path + ".weights", 'wb') as fd:
                pickle.dump(weights, fd)

    def dump_libsvm(self, path: str, dump_weights: bool = True) -> None:
        """ Dump the collected data to the specified path as a libsvm file """
        sparse = self.retrieve_sparse_matrix()
        labels = self.retrieve_labels()
        dump_svmlight_file(sparse, labels, path)

        feature_names = self.retrieve_feature_names_as_list()
        with open(path + ".feature_names", 'wb') as fd:
            pickle.dump(feature_names, fd)

        if dump_weights and len(Counter(labels).keys()) == 4:
            weights = self.retrieve_label_weights(num_labels=4)
            with open(path + ".weights", 'wb') as fd:
                pickle.dump(weights, fd)

    def retrieve_debug_output(self) -> List[Dict[str, float]]:
        """
        Retrieve JSON pretty printed data to verify that the features are transformed correctly.
        """
        feature_names: List[str] = self.retrieve_feature_names_as_list()
        csr_mat = self.retrieve_sparse_matrix()
        matrix = csr_mat.todense()
        assert matrix.shape[1] <= len(feature_names), f"Number of columns exceeds number of features: Matrix: {matrix.shape[1]} -- Features: {len(feature_names)}"

        numerical_dict_features: List[Dict[str, float]] = list()
        for i in range(matrix.shape[0]):
            numerical_dict_features.append(dict())
            for j in range(matrix.shape[1]):
                numerical_dict_features[i][feature_names[j]] = matrix[i, j]

        return numerical_dict_features

    def print_feature_info(self) -> None:
        """Output information on the features """
        logger.info(f"Number of Per-Pixel Features: {self.num_pixel_features}")
        logger.info(f"Number of Features Total: {self.num_features}")

    def dump_feature_map(self, filename: str) -> None:
        """
        Produces a named feature map for use with XGBoost.
        :param filename: feature map filename
        """
        with open(filename, 'w') as fd:
            flist = self.retrieve_feature_names_as_list()
            for f in flist:
                fd.write(f + "\n")
        logger.info(f"Extracted xgboost feature map to {filename}")

    def extract_features(self, input_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Extract pixel data from given input dictionary and apply featue extraction methods.
        Intended for transforming test data for classification.
        :param input_data: pixel data to transform
        """
        for entry_name, entry_values in input_data.items():
            for feature in self.feature_mapping["per_pixel_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Defined per-pixel function not found: {feature['function']}"
                    getattr(self, feature["function"])(entry_values, **feature["args"])
                    self._increment_col(feature["vector_size"])

            # before moving to the next cookie entry, reset the column index and move to the next row
            self._reset_col()
            self._increment_row()

    def extract_features_with_labels(self, input_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Intended for the training data feature extraction. Expects labels in the input dictionary.
        Filters unwanted labels
        Performs timing measurements to analyze the feature extraction performance.
        :param input_data: Pixel training data to transform, with labels.
        """
        timings_per_function: Dict[str, List] = dict()

        ctr_label_skipped: int = 0

        logger.info("Begin feature extraction process...")
        start = time.perf_counter()
        for entry_name, entry_values in tqdm(input_data.items()):
            # retrieve the label and skip ones we don't want
            category_label = int(entry_values["label"])

            # Make sure we only consider desired labels
            if not (0 <= category_label <= 3):
                ctr_label_skipped += 1
                continue

            # append the label to the list
            self._insert_label(category_label)

            # Extract features from cookie data that is consistent across all updates.
            # This includes name, domain, path and first-party domain
            for feature in self.feature_mapping["per_pixel_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Defined per-pixel function not found: {feature['function']}"

                    if feature["function"] not in timings_per_function:
                        timings_per_function[feature["function"]] = list()

                    function = getattr(self, feature["function"])

                    t_start = time.perf_counter_ns()
                    function(entry_values, **feature["args"])
                    timings_per_function[feature["function"]].append(time.perf_counter_ns() - t_start)

                    self._increment_col(feature["vector_size"])

            # before moving to the next cookie entry, reset the column index and move to the next row
            self._reset_col()
            self._increment_row()

        end = time.perf_counter()
        total_time_taken: float = end - start
        logger.info(f"Feature extraction completed. Final row position: {self._current_row}")

        logger.info("Timings per feature:")
        total_time_spent = 0
        for func, t_list in sorted(timings_per_function.items(), key=lambda x: sum(x[1]), reverse=True):
            if len(t_list) == 0:
                continue
            else:
                time_spent = sum(t_list)
                total_time_spent += time_spent
                logmsg = (f"total:{sum(t_list) / 1e9:.3f} s"
                          f"|{sum(t_list) / (1e7 * total_time_taken):.3f}%"
                          f"|mean: {mean(t_list):.2f} ns|max: {max(t_list)} ns")
                if len(t_list) >= 2:
                    logmsg += f"|stdev: {stdev(t_list):.2f} ns"
                logmsg += f"|{func}"
                logger.info(logmsg)
        logger.info(f"Total time spent in feature extraction: {total_time_spent / 1e9:.3f} seconds")
        logger.info(f"Time lost to overhead: {total_time_taken - (total_time_spent / 1e9):.3f} seconds")
        logger.info(f"Num social media category skipped: {ctr_label_skipped}")

    
    #
    ## Setup methods for external resources
    ## TODO add more to add features
    #

    def setup_top_domains(self, source: str, vector_size: int) -> None:
        """
        Sets up the lookup table to determine if and on which rank of
        the top k domains from the external source ranking our queried domain is.
        The source ranking is assumed to be sorted in advance.
        :param source: Path to source ranking
        :param vector_size: How many top domains to include in the lookup table. i.e., k
        """
        self._top_domain_lookup = load_lookup_from_csv(source, vector_size)

    def setup_top_query_param(self, source: str, vector_size: int) -> None:
        """
        Sets up the lookup table to check which of the k top query parameters are present. 
        The source ranking is assumed to be sorted in advance.
        :param source: path to source ranking
        :param vector_size: How many top query parameters to include in the lookup table, i.e., k
        """
        self._top_query_param_lookup = load_lookup_from_csv(source, vector_size)

    def setup_top_path_piece(self, source: str, vector_size: int) -> None:
        """
        Sets up the lookup table to check which of the k most common pieces in a path are present in the url.
        The source ranking is assumed to be sorted in advance.
        :param source: path to source ranking
        :param vector_size: How many (k) top path pieces to include in the lookup table.
        """
        self._top_path_piece_lookup = load_lookup_from_csv(source, vector_size)

    def setup_top_t_o_domain(self, source: str, vector_size: int) -> None:
        """
        Same as setup_top_domains but the external ranking is from triggering origin urls not request urls
        :param source: path to source ranking
        :param vector_size: How many top domains to include in the lookup table, i.e., k
        """
        self._top_t_o_domain_lookup = load_lookup_from_csv(source, vector_size)
    
    def setup_mode(self, source: str, vector_size: int) -> None:
        """
        Sets up the lookup table for image colour mode from an external source. The source
        is not sorted.
        :param source: path to source
        :param vector_size: Number of different image colour modes to include (length of external resource).
        """
        self._mode_lookup = load_lookup_from_csv(source, vector_size)

    def setup_format(self, source: str, vector_size: int) -> None:
        """
        Sets up the lookup table for image format from an external source. The source is not sorted.
        :param source: path to source
        :param vector_size: Number of different image formats to include
        """
        self._format_lookup = load_lookup_from_csv(source, vector_size)
    #
    ## Per pixel features
    ## TODO add more
    #

    def feature_top_domains(self, pixel_features: Dict[str, Any]) -> None:
        """
        This feature function detects whether the pixel domain is part of the top K pixel domains
        from the external resource document, and constructs a K-sized one-hot vector feature.
        :param pixel_features: Dictionary containing key "url" of the pixel
        """
        assert (self._top_domain_lookup is not None), "Top N domain lookup was not set up prior to feature extraction!"
        pixel_domain: str = url_parse_to_uniform(pixel_features["url"])
        if pixel_domain in self._top_domain_lookup:
            rank = self._top_domain_lookup[pixel_domain]
            self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_top_query_param(self, pixel_features: Dict[str, Any]) -> None:
        """
        This function detects whether the url of the pixel contains (1.0) or does not
        contain (0.0) each of the N most common query parameters from the external resource document
        :param pixel_features: Dictionary containing key "url" of the pixel
        """
        assert (self._top_query_param_lookup is not None), "Top N query parameters in url was not set up prior to feature extraction!"
        
        obj = parse.urlsplit(pixel_features["url"])
        q_dict = parse.parse_qs(obj.query)
        keys = q_dict.keys()
        for k in keys:
            if k in self._top_query_param_lookup:
                rank = self._top_query_param_lookup[k]
                self._insert_sparse_entry(1.0, col_offset = rank)

    def feature_has_query_param(self, pixel_features: Dict[str, Any]) -> None:
        obj = parse.urlsplit(pixel_features["url"])
        if (len(obj) > 0):
            self._insert_sparse_entry(1.0)

    def feature_top_path_piece(self, pixel_features: Dict[str, Any]) -> None:
        """
        detects whether or not the path of a given url contains (1.0) or not (0.0) each of the N most common
        'words' in the path.
        :param pixel_features: Dictionary containing key "url" of the pixel
        """
        assert (self._top_path_piece_lookup is not None), "Top N path pieces in url was not set up prior to feature extraction!"
        obj = parse.urlsplit(pixel_features["url"])
        url_path = obj.path.split("/")
        for p in url_path:
            if p in self._top_path_piece_lookup:
                rank = self._top_path_piece_lookup[p]
                self._insert_sparse_entry(1.0, col_offset = rank)

    def feature_is_third_party(self, pixel_features: Dict[str, Any]) -> None:
        """
        single feature entry, inserts 1.0 if the domain of the website loading the pixel and the domain of
        the pixel (from its url) are not the same
        :param pixel_features: Dictionary containing keys "domain" and "first_party_domain"
        """
        pixel_domain = url_to_uniform_domain(parse.urlsplit(pixel_features["url"]).netloc)
        website_domain = url_to_uniform_domain(parse.urlsplit(pixel_features["first_party_domain"]).netloc)
        if pixel_domain not in website_domain:
            self._insert_sparse_entry(1.0)

    def feature_top_t_o_domain(self, pixel_features: Dict[str, Any]) -> None:
        assert (self._top_t_o_domain_lookup is not None), "Top N triggering origin domain lookup was not set up prior to feature extraction!"
        t_o_domain: str = url_parse_to_uniform(pixel_features["triggering_origin"])
        if t_o_domain in self._top_t_o_domain_lookup:
            rank = self._top_t_o_domain_lookup[t_o_domain]
            self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_shannon_entropy_url(self, pixel_features: Dict[str, Any]) -> None:
        content_char_counts = Counter([ch for ch in pixel_features["url"]])
        total_string_size = len(pixel_features["url"])
        
        entropy: float = 0
        for ratio in [char_count / total_string_size for char_count in content_char_counts.values()]:
            entropy -= ratio * log(ratio, 2)

        self._insert_sparse_entry(entropy)

    def feature_shannon_entropy_headers(self, pixel_features: Dict[str, Any]) -> None:
        content_char_counts = Counter([ch for ch in pixel_features["headers"]])
        total_string_size = len(pixel_features["headers"])

        entropy: float = 0
        for ratio in [char_count / total_string_size for char_count in content_char_counts.values()]:
            entropy -= ratio * log(ratio, 2)

        self._insert_sparse_entry(entropy)

    def feature_is_1x1(self, pixel_features: Dict[str, Any]) -> None:
        size = pixel_features["img_size"]
        if size[0] == 1 and  size[1]  == 1:
            self._insert_sparse_entry(1.0)

    def feature_transparency(self, pixel_features: Dict[str, Any]) -> None:
        alpha = pixel_features["img_colour"][3]
        self._insert_sparse_entry(alpha)

    def feature_colour(self, pixel_features: Dict[str, Any]) -> None:
        colour = pixel_features["img_colour"]
        for i in range(3):
            self._insert_sparse_entry(colour[i], col_offset=i)

    def feature_format(self, pixel_features: Dict[str, Any]) -> None:
        assert (self._format_lookup is not None), "format lookup was not set up prior to feature extraction"
        img_format = pixel_features["img_format"]
        if img_format in self._format_lookup:
            rank = self._format_lookup[img_format]
            self._insert_sparse_entry(1.0, col_offset = rank)

    def feature_mode(self, pixel_features: Dict[str, Any]) -> None:
        assert (self._mode_lookup is not None), "mode lookup was not set up prior to feature extraction"
        img_mode = pixel_features["img_mode"]
        if img_mode in self._mode_lookup:
            rank = self._mode_lookup[img_mode]
            self._insert_sparse_entry(1.0, col_offset = rank)

    def feature_size(self, pixel_features: Dict[str, Any]) -> None:
        size = pixel_features["img_size"]
        self._insert_sparse_entry(size[0])
        self._insert_sparse_entry(size[1], col_offset=1)

    def feature_url_length(self, pixel_features: Dict[str, Any]) -> None:
        obj = parse.urlsplit(pixel_features["url"])
        self._insert_sparse_entry(len(obj.path), col_offset=0)
        self._insert_sparse_entry(len(obj.query), col_offset=1)
        self._insert_sparse_entry(len(parse.parse_qs(obj.query)), col_offset=2)

    def feature_header_length(self, pixel_features: Dict[str, Any]) -> None:
        self._insert_sparse_entry(len(pixel_features["headers"]))

    def feature_header_fields(self, pixel_features: Dict[str, Any]) -> None:
        headers = pixel_features["headers"] #headers is a string
        h2 = headers[2:-2] #split of [[ and ]] at beginning and end
        h3 = h2.split("],[") #split into fields and their values
        for field in h3:
            field_name = field.split('","')[0][1:]
            if field_name == "Cookie":
                self._insert_sparse_entry(1.0, col_offset = 0)
            elif field_name == "Referer":
                self._insert_sparse_entry(1.0, col_offset = 1)
            elif field_name == "Origin":
                self._insert_sparse_entry(1.0, col_offset = 2)
            elif field_name == "Alt-Used":
                self._insert_sparse_entry(1.0, col_offset = 3)

    def feature_is_blocked(self, pixel_features: Dict[str, Any]) -> None:

        self._insert_sparse_entry(pixel_features["blocked"][0], col_offset=0) #EasyPrivacy
        self._insert_sparse_entry(pixel_features["blocked"][1], col_offset=1) #EasyList

    def feature_compressed_url(self, pixel_features: Dict[str, Any]) -> None:
        """
        Number of bytes of the compressed content using zlib, as well as size reduction.
        This serves as a heuristic to represent entropy. If entropy is high, then the compressed
        data will like have around the same size as the uncompressed data. High entropy data is
        likely to be a randomly generated string. Low entropy data will have a stronger reduction
        in size after compression.
        :param pixel_features: Dictionary containing key "url".
        """
        unquoted_content = parse.unquote(pixel_features["url"])
        content_bytes = bytes(unquoted_content.encode("utf-8"))
        compressed_size = len(zlib.compress(content_bytes, level=9))

        # Append compressed size
        self._insert_sparse_entry(compressed_size, col_offset=0)

        # Append reduction
        reduced = len(content_bytes) - compressed_size
        self._insert_sparse_entry(reduced, col_offset=1)

    def feature_compressed_headers(self, pixel_features: Dict[str, Any]) -> None:
        """
        Number of bytes of the compressed content using zlib, as well as size reduction.
        This serves as a heuristic to represent entropy. If entropy is high, then the compressed
        data will like have around the same size as the uncompressed data. High entropy data is
        likely to be a randomly generated string. Low entropy data will have a stronger reduction
        in size after compression.
        :param pixel_features: Dictionary containing key "headers".
        """
        unquoted_content = parse.unquote(pixel_features["headers"])
        content_bytes = bytes(unquoted_content.encode("utf-8"))
        compressed_size = len(zlib.compress(content_bytes, level=9))

        # Append compressed size
        self._insert_sparse_entry(compressed_size, col_offset=0)

        # Append reduction
        reduced = len(content_bytes) - compressed_size
        self._insert_sparse_entry(reduced, col_offset=1)

