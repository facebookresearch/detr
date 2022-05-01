# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

# from .coco import build as build_coco
from dataset import LabeledDataset
import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    # if args.dataset_file == 'coco':
    #     return build_coco(image_set, args)
    # if args.dataset_file == 'coco_panoptic':
    #     # to avoid making panopticapi required for coco
    #     from .coco_panoptic import build as build_coco_panoptic
    #     return build_coco_panoptic(image_set, args)
    # raise ValueError(f'dataset {args.dataset_file} not supported')
    if image_set == 'train':
        return LabeledDataset(root="/scratch/yk1962/datasets/labeled_data", split="training", transforms=get_transform(train=True))
    else:
        return LabeledDataset(root="/scratch/yk1962/datasets/labeled_data", split="validation", transforms=get_transform(train=False))