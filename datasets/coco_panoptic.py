# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms


class CocoPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')
        ann_path = Path(self.ann_folder) / ann_info['file_name']

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco['images'])

    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width


def build(image_set, args):
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
    mode = 'panoptic'
    PATHS = {
        "train": ("train2017", Path("annotations") / f'{mode}_train2017.json'),
        "val": ("val2017", Path("annotations") / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder = ann_folder_root / f'{mode}_{img_folder}'
    ann_file = ann_folder_root / ann_file

    dataset = CocoPanoptic(img_folder_path, ann_folder, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset
