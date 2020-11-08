"""
This downloads the Food dataset from the AICrowd servers and
does the pre-processing and fixes some issues with the data
"""

import os
import json
import functools
import pathlib
import shutil
import requests

import tqdm
import cv2 as cv


def __download_file(url, filename):
    """
    Downloads a file with progress bar
    :param url: link to download from
    :param filename: the file to save to
    """
    req = requests.get(url, stream=True, allow_redirects=True)
    if req.status_code != 200:
        req.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {req.status_code}")
    file_size = int(req.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    req.raw.read = functools.partial(req.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.tqdm.wrapattr(req.raw, "read", total=file_size,
                            desc="Downloading %s"%(filename.split('/')[-1])) as r_raw:
        with path.open("wb") as file_obj:
            shutil.copyfileobj(r_raw, file_obj)

    return path


def download_dataset(working_dir):
    """
    Downloads the dataset, unzips it and puts in it in the right location
    """
    base_url = "https://datasets.aicrowd.com/default/aicrowd-public-datasets/" + \
               "food-recognition-challenge/v0.4/"
    if not os.path.exists(os.path.join(working_dir, 'data', 'train.zip')):
        __download_file(base_url + "train-v0.4.tar.gz",
                        os.path.join(working_dir, 'data', 'train.zip'))
    if not os.path.exists(os.path.join(working_dir, 'data', 'val.zip')):
        __download_file(base_url + "val-v0.4.tar.gz",
                        os.path.join(working_dir, 'data', 'val.zip'))
    if not os.path.exists(os.path.join(working_dir, 'data', 'test.zip')):
        __download_file(base_url + "test_images-v0.4.tar.gz",
                        os.path.join(working_dir, 'data', 'test.zip'))

    if not os.path.exists(os.path.join(working_dir, 'data', 'train')):
        os.system('unzip -q %s -d %s'%(os.path.join(working_dir, 'data', 'train.zip'),
                                       os.path.join(working_dir, 'data')))
        print('Unzipping Training Dataset is complete')
    if not (os.path.exists(os.path.join(working_dir, 'data', 'test')) and \
            os.path.exists(os.path.join(working_dir, 'data', 'val'))):
        os.system('unzip -q %s -d %s'%(os.path.join(working_dir, 'data', 'test.zip'),
                                       os.path.join(working_dir, 'data')))
        os.system('mv %s %s'%(os.path.join(working_dir, 'data', 'val'),
                              os.path.join(working_dir, 'data', 'test')))
        print('Unzipping Test Dataset is complete')
        os.system('unzip -q %s -d %s'%(os.path.join(working_dir, 'data', 'val.zip'),
                                       os.path.join(working_dir, 'data')))
        print('Unzipping Validation Dataset is complete')


def print_data_description(working_dir):
    """
    Corrects the annotations and prints a few details about the dataset
    :param working_dir: str, the images are expected in
                        working_dir/data/<train-test-val>/annotations.json
    """
    print('Train images from train folder directory:',
          len(os.listdir(os.path.join(working_dir, 'data', 'train', 'images'))))
    print('Validation images from val folder directory:',
          len(os.listdir(os.path.join(working_dir, 'data', 'val', 'images'))))
    print('Test images from val folder directory:',
          len(os.listdir(os.path.join(working_dir, 'data', 'test', 'images'))))

    with open(os.path.join(working_dir, 'data', 'train', 'annotations.json')) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(working_dir, 'data', 'val', 'annotations.json')) as json_file:
        val_json = json.load(json_file)
    with open(os.path.join(working_dir, 'data', 'test', 'annotations.json')) as json_file:
        test_json = json.load(json_file)

    print('Keys of train annotation file:', train_json.keys())
    print('No of Categories(Classes) present:', len(train_json['categories']))
    print('Example:', train_json['categories'][0])
    print('Keys of val annotation file:', val_json.keys())
    print('No of Categories(Classes) present:', len(val_json['categories']))
    print('Example:', val_json['categories'][0])
    print('Keys of test annotation file:', test_json.keys())
    print('No of Categories(Classes) present:', len(test_json['categories']))
    print('Example:', test_json['categories'][0])


def fix_errors(working_dir, directory):
    """
    Fixes the mislabelled image sizes
    :param working_dir: str, the images are expected in
                        working_dir/data/<train-test-val>/images/*.jpg
    :param directory: str, one of train, test, or val
    """
    errors_fixed = 0
    with open(os.path.join(working_dir, 'data', directory, 'annotations.json')) as json_file:
        data_json = json.load(json_file)
    image_list = []
    with tqdm.tqdm(data_json['images']) as progress_bar:
        for record in progress_bar:
            img = cv.imread(os.path.join(working_dir, 'data', directory,
                                         'images', record['file_name']))
            if record['height'] != img.shape[0] or record['width'] != img.shape[1]:
                errors_fixed += 1
                progress_bar.set_postfix(fixed=errors_fixed)
            else:
                image_list.append(record)
        data_json['images'] = image_list
    with open(os.path.join(working_dir, 'data', directory, 'annotations.json'), 'w') as outfile:
        json.dump(data_json, outfile)
