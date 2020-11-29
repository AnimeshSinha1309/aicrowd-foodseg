import os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tqdm 


working_dir = 'scratch'
subset_dirs = ['train', 'val', 'test']

for subset_dir in subset_dirs[1:]:
    read_dir = os.path.join(working_dir, 'data', subset_dir)
    data_csv = {'image': [], 'class': [], 'width': [], 'height': []}
    data = json.load(open(os.path.join(read_dir, 'annotations.json')))
    image_to_filename = {image['id']: image['file_name'] for image in data['images']}
    category_map = {cat['id']: cat['name'] for cat in data['categories']}
    os.makedirs(os.path.join(working_dir, 'classify', subset_dir), exist_ok=True)
    for annotation in tqdm.tqdm(data['annotations']):
        image_filename = image_to_filename[annotation['image_id']]
        image_name = os.path.join(read_dir, 'images', image_filename)
        image = plt.imread(image_name)
        x, y, w, h = list(map(int, annotation['bbox']))
        result = image[x:x+w, y:y+h, :]
        write_to = os.path.join(working_dir, 'classify', subset_dir, image_filename)
        data_csv['image'].append(image_filename)
        data_csv['class'].append(category_map[annotation['category_id']])
        data_csv['width'].append(w)
        data_csv['height'].append(h)
        plt.imsave(write_to, result)
    pd.DataFrame(data_csv).to_csv(os.path.join(working_dir, 'classify', subset_dir, 'data.csv'), index=False)
