'''
Used to convert annotations made by blender module into coco format.
'''

import os
import pandas as pd
import json
from datetime import datetime

file_location = os.path.dirname(__file__)
os.chdir(file_location)

#need to convert to correct label terminology using object dict
object_params = pd.read_json('src/object_dict.json')
label_cats = [object_params[blend_obj]['label_cat'] for blend_obj in object_params]
label_cats_unique = list(dict.fromkeys(label_cats))#drops duplicates

#Reading in blender generated annotations
raw_annotations = pd.read_csv('annotations.tsv', sep='\t', header=0)
#replacing full filepath with image name
filenames = [os.path.basename(file_path) for file_path in raw_annotations['file_path']]
raw_annotations['file_path'] = filenames
#replacing items with label cats
raw_annotations['object_name'] = [object_params[blend_obj]['label_cat'] for blend_obj in raw_annotations['object_name']]

def bbox_convert(top_left_xy, right_bottom_xy):
    """
    Take blender annotation bbox output, convert to coco format.
    Returns coco formatted bbox and area
    """ 
    x_ul, y_ul = eval(top_left_xy)
    x_br, y_br = eval(right_bottom_xy)
    width = x_br - x_ul
    height = y_ul - y_br
    area = width*height
    return [x_ul, y_ul, width, height], area

#Starting dict creation
info = {
    'description': 'Blender generated image annotations',
    'dataset_version': '1',
    'date_created': datetime.today().strftime('%Y-%m-%d')
}

licenses = [
    {
            'id': 1,
            'url': '',
            'name': 'Fellowship.ai'
    }
]

images = []
image_lookup_dict = {}#easier to access than list of dicts that coco format uses
filenames_unique = list(dict.fromkeys(filenames))
for i, image in enumerate(filenames_unique):
    temp_dict = {
        'id': i,
        'width': 400,
        'height': 600,
        'filename': image,
        'license': 0,
        'flickr_url': '',
        'coco_url': '',
        'date_captured': ''
    }
    images.append(temp_dict)
    image_lookup_dict[image] = i

categories = []
category_lookup_dict = {}
for i, label in enumerate(label_cats):
    temp_dict = {
        'id': i+1,#other annotation files start with 1 index
        'name': label_cats[i],
        'supercategory': 'N/A'
    }
    categories.append(temp_dict)
    category_lookup_dict[label_cats[i]] = i+1

annotations = []
for i, anno in enumerate(raw_annotations):
    bbox, area = bbox_convert(raw_annotations['top_left_xy'][i], raw_annotations['right_bottom_xy'][i])
    temp_dict = {
        'id': i,
        'image_id': image_lookup_dict[raw_annotations['file_path'][i]],
        'category_id': category_lookup_dict[raw_annotations['object_name'][i]],
        'bbox': bbox,
        'area': area,
        'segmentation': [],
        'iscrowd': 0
    }
    annotations.append(temp_dict)

coco_formatted = {
    'info': info,
    'licenses': licenses,
    'images': images,
    'annotations': annotations,
    'categories': categories
}

with open('synthetic_annotations_coco.json','w') as f:
    json.dump(coco_formatted,f)