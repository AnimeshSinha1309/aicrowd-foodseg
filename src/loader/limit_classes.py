"""
Creates annotation files to train on fewer classes than all of them
to simplify the task / reduce computation required.
"""

import os
import json
import numpy as np


def limit_classes(target_json, classes_taken):
    """
    We are training on top-k classes in the train, so this class_frequencies
    value should not be recomputed for validation / test.
    """
    result = dict()
    for key, value in target_json.items():
        if key not in ['annotations', 'classes']:
            result[key] = value
    result['annotations'] = list(filter(
        lambda x: x['category_id'] in classes_taken, target_json['annotations']))
    result['categories'] = list(filter(lambda x: x['id'] in classes_taken,
                                       target_json['categories']))
    return result


def limit_files(working_dir, directory_train, directory_targets, classes_count):
    """
    Actually rewrites the file
    """
    # Generate the List of classes from training file
    with open(os.path.join(working_dir, 'data', directory_train,
                           'annotations.json'), 'r') as json_file:
        train_json = json.load(json_file)
    class_labels, class_counts = np.unique(list(map(
        lambda x: x['category_id'], train_json['annotations'])),
        return_counts=True)
    class_frequencies = {label: count for count, label in sorted(
        list(zip(class_counts, class_labels)), reverse=True)}
    classes_taken = list(class_frequencies.keys())[:classes_count]
    # Loop over all targets and fix them
    for directory_target in directory_targets:
        with open(os.path.join(working_dir, 'data', directory_target,
                               'annotations.json'), 'r') as json_file:
            data_json = json.load(json_file)
        data_json = limit_classes(data_json, classes_taken)
        print(len(data_json['annotations']),
            'annotations with', len(data_json['categories']),
            'categories.')
        with open(os.path.join(working_dir, 'data', directory_target,
                               'processed_annotations.json'), 'w') as outfile:
            json.dump(data_json, outfile)
