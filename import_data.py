from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import Data_set


defaults = collections.OrderedDict([
    ("c0", [0]),
    ("c1", [0]),
    ("c2", [0]),
    ("c3", [0]),
    ("c4", [0]),
    ("c5", [0]),
    ("c6", [0]),
    ("c7", [0]),
    ("c8", [0]),
    ("c9", [0]),
    ("c10",[0]),
    ("fitnesses",[0.0])
])  

def dataset(y_name="fitnesses", train_fraction=0.7):

    # data = Data_set.load_data()
    filename = 'f_and_p.csv'
    # data.to_csv(filename,index=False)

    def decode_line(line):
        items = tf.decode_csv(line, list(defaults.values()))
        pairs = zip(defaults.keys(), items)
        features_dict = dict(pairs)
        label = features_dict.pop(y_name)
        return features_dict, label

    def in_training_set(line):
        num_buckets = 1000000
        bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
        return bucket_id < int(train_fraction * num_buckets)

    def in_eval_set(line):
        return ~in_training_set(line)

    base_dataset = tf.data.TextLineDataset(filename)

    train = (base_dataset
            # Take only the training-set lines.
            .filter(in_training_set)
            # Decode each line into a (features_dict, label) pair.
            .map(decode_line)
            # Cache data so you only decode the file once.
            .cache())
    # Do the same for the test-set.
    eval = (base_dataset.filter(in_eval_set).map(decode_line).cache())

    return train, eval