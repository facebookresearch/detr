# Please do not change this file.
# We will use this file to benchmark your model.
# If you find a bug, post it on campuswire.

import os
import yaml
import random
from PIL import Image
import torch

class_dict = {
    'cup or mug': 0,
    'bird': 1,
    'hat with a wide brim': 2,
    'person': 3,
    'dog': 4,
    'lizard': 5,
    'sheep': 6,
    'wine bottle': 7,
    'bowl': 8,
    'airplane': 9,
    'domestic cat': 10,
    'car': 11,
    'porcupine': 12,
    'bear': 13,
    'tape player': 14,
    'ray': 15,
    'laptop': 16,
    'zebra': 17,
    'computer keyboard': 18,
    'pitcher': 19,
    'artichoke': 20,
    'tv or monitor': 21,
    'table': 22,
    'chair': 23,
    'helmet': 24,
    'traffic light': 25,
    'red panda': 26,
    'sunglasses': 27,
    'lamp': 28,
    'bicycle': 29,
    'backpack': 30,
    'mushroom': 31,
    'fox': 32,
    'otter': 33,
    'guitar': 34,
    'microphone': 35,
    'strawberry': 36,
    'stove': 37,
    'violin': 38,
    'bookshelf': 39,
    'sofa': 40,
    'bell pepper': 41,
    'bagel': 42,
    'lemon': 43,
    'orange': 44,
    'bench': 45,
    'piano': 46,
    'flower pot': 47,
    'butterfly': 48,
    'purse': 49,
    'pomegranate': 50,
    'train': 51,
    'drum': 52,
    'hippopotamus': 53,
    'ski': 54,
    'ladybug': 55,
    'banana': 56,
    'monkey': 57,
    'bus': 58,
    'miniskirt': 59,
    'camel': 60,
    'cream': 61,
    'lobster': 62,
    'seal': 63,
    'horse': 64,
    'cart': 65,
    'elephant': 66,
    'snake': 67,
    'fig': 68,
    'watercraft': 69,
    'apple': 70,
    'antelope': 71,
    'cattle': 72,
    'whale': 73,
    'coffee maker': 74,
    'baby bed': 75,
    'frog': 76,
    'bathing cap': 77,
    'crutch': 78,
    'koala bear': 79,
    'tie': 80,
    'dumbbell': 81,
    'tiger': 82,
    'dragonfly': 83,
    'goldfish': 84,
    'cucumber': 85,
    'turtle': 86,
    'harp': 87,
    'jellyfish': 88,
    'swine': 89,
    'pretzel': 90,
    'motorcycle': 91,
    'beaker': 92,
    'rabbit': 93,
    'nail': 94,
    'axe': 95,
    'salt or pepper shaker': 96,
    'croquet ball': 97,
    'skunk': 98,
    'starfish': 99
}


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.transform = transform

        self.image_dir = root
        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        with open(os.path.join(self.image_dir, f"{idx}.PNG"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img)


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /labeled
            split: The split you want to used, it should be training or validation
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transforms = transforms

        self.image_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")

        self.num_images = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.num_images  # self.num_images

    def __getitem__(self, idx):
        # the idx of training image is from 1 to 30000
        # the idx of validation image is from 30001 to 50000

        if self.split == "training":
            offset = 1
        if self.split == "validation":
            offset = 30001

        with open(os.path.join(self.image_dir, f"{idx + offset}.JPEG"), 'rb') as f:
            img = Image.open(f).convert('RGB')
        with open(os.path.join(self.label_dir, f"{idx + offset}.yml"), 'rb') as f:
            yamlfile = yaml.load(f, Loader=yaml.FullLoader)

        num_objs = len(yamlfile['labels'])
        # xmin, ymin, xmax, ymax
        boxes = torch.as_tensor(yamlfile['bboxes'], dtype=torch.float32)
        labels = []
        for label in yamlfile['labels']:
            labels.append(class_dict[label])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
