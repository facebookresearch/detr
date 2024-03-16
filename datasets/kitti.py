"""
KITTI dataset ckass for DeTR
"""
import os
import os.path as osp
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import datasets.transforms as T

SPLIT = ['train', 'val', 'test']

class KITTIDataset(Dataset):
    def __init__(self, base_path = '/srip-vol/datasets/KITTI3D', split = 'train', transform = None):
        assert split in SPLIT
        self.split = split
        self.base_path = base_path
        self.isTest = self.split == 'test'
        self.folder_name = 'testing' if self.split == 'test' else 'training'

        # Read imageset with index
        image_set_path = osp.join(self.base_path, 'ImageSets', self.split + '.txt')
        lines = open(image_set_path).readlines()
        self.image_set = [line.strip() for line in lines]
        # Define transform
        self._transforms = transform
        # Set-up paths
        self.image_path = osp.join(base_path,  self.folder_name, 'image_2')
        if not self.isTest:
            self.label_path = osp.join(base_path,  self.folder_name, 'label_2')

        self.KITTI_CLASS = ['Car', 'Pedestrian', 'Cyclist']
        self.prepare = ConvertCocoPolysToMask()

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        '''
        Return a dict with following fields
        'image' - image as a numpy array
        'label' - list of dicts each with label info parsed
        '''
        data_idx = self.image_set[idx]
        data = {}
        # img = np.asarray(Image.open(osp.join(self.image_path, data_idx + '.png')))
        img = Image.open(osp.join(self.image_path, data_idx + '.png'))
        if not self.isTest:
            label = self.__read_label_data(data_idx)

        bbox_data = []
        class_data = []
        for i in range(len(label)):
            bbox_data.append(label[i]['bbox_2d'])
            class_data.append(label[i]['class_id'])

        target = {}
        target['boxes'] = torch.as_tensor(bbox_data)
        target['labels'] = torch.as_tensor(class_data)
        target['image_id'] = int(data_idx)

        # Prepare dataset
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __read_label_data(self, idx):
        '''
        Function to read label data from text file
        '''
        lines = open(osp.join(self.label_path, idx + '.txt')).readlines()
        label = []
        for line in lines:
            data = line.split(' ')
            if data[0] in self.KITTI_CLASS:
                label.append(KITTI_label(data[0],
                    float(data[1]) , float(data[2]) , float(data[3]),
                    float(data[4]) , float(data[5]) , float(data[6]),
                    float(data[7]) , float(data[8]) , float(data[9]),
                    float(data[10]), float(data[11]), float(data[12]),
                    float(data[13]), float(data[14])))
        return label


KITTI_CLASS = {'Car': 1, 'Pedestrian': 2, 'Cyclist' : 3}

def KITTI_label(class_name, truncated, occluded, alpha,
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
    dim_h, dim_w, dim_l, x_c, y_c, z_c, rot_y, score = 0):
    '''
    To create a label dict
    Note - score field added at last with default val 0
    '''
    label_info = {}
    label_info['class_id'] = KITTI_CLASS[class_name]
    label_info['truncated'] = truncated
    label_info['occluded'] = occluded
    label_info['alpha'] = alpha
    # label_info['bbox_2d'] = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
    label_info['bbox_2d'] = [(bbox_xmax + bbox_xmin)/2, (bbox_ymax + bbox_ymin)/2, bbox_xmax-bbox_xmin, bbox_ymax-bbox_ymin]
    label_info['dim'] = [dim_h, dim_w, dim_l]
    label_info['loc'] = [x_c, y_c, z_c]
    label_info['rot_y'] = rot_y
    label_info['score'] = score

    return label_info

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class ConvertCocoPolysToMask(object):
    # def __init__(self, return_masks=False):
    #     self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        # anno = target["annotations"]
        # anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = target['boxes']#torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # classes = [obj["category_id"] for obj in anno]
        classes = target['labels']# torch.tensor(classes, dtype=torch.int64)

        # if self.return_masks:
            # segmentations = [obj["segmentation"] for obj in anno]
            # masks = convert_coco_poly_to_mask(segmentations, h, w)

        # keypoints = None
        # if anno and "keypoints" in anno[0]:
        #     keypoints = [obj["keypoints"] for obj in anno]
        #     keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        #     num_keypoints = keypoints.shape[0]
        #     if num_keypoints:
        #         # keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        # classes = classes[keep]
        # if self.return_masks:
        #     masks = masks[keep]
        # if keypoints is not None:
        #     keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        # if self.return_masks:
        #     target["masks"] = masks
        # target["image_id"] = image_id
        # if keypoints is not None:
        #     target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = [] #area[keep]
        # target["iscrowd"] = [] #iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def build(image_set, args):
    # base_path = '/srip-vol/datasets/KITTI3D'
    dataset = KITTIDataset(base_path = args.kitti_path, split = image_set, transform = make_coco_transforms(image_set))
    return dataset 