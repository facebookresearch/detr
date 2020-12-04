import argparse
import datetime
import json

import time
from pathlib import Path

import cv2
from torch.utils.data import DataLoader, DistributedSampler
import random
import numpy as np
import torch
from PIL import Image
import util.misc as utils
import datasets

from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch
from models import build_model

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.detr import DETR
from torchvision import transforms
import matplotlib.pyplot as plt

# from datasets import transforms


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser



category_filter = [
'person',
'bicycle',
'car',
'motorcycle',
'bus',
'train',
'truck',
'boat',

]
categories = [
'unlabeled',
'person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'backpack',
'umbrella',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'dining table',
'toilet',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush']


cate_to_colors = {
'unlabeled': (0, 255, 0),
'person': (0, 150, 255),
'bicycle': (255, 128, 0),
'car': (255, 153, 255),
'motorcycle': (255, 153, 255),
'airplane': (0, 255, 0),
'bus': (255, 153, 255),
'train': (255, 153, 255),
'truck': (255, 153, 255),
'boat': (0, 255, 0),
'traffic light': (0, 255, 0),
'fire hydrant': (0, 255, 0),
'stop sign': (0, 255, 0),
'parking meter': (0, 255, 0),
'bench': (0, 255, 0),
'bird': (0, 255, 0),
'cat': (0, 255, 0),
'dog': (0, 255, 0),
'horse': (0, 255, 0),
'sheep': (0, 255, 0),
'cow': (0, 255, 0),
'elephant': (0, 255, 0),
'bear': (0, 255, 0),
'zebra': (0, 255, 0),
'giraffe': (0, 255, 0),
'backpack': (0, 255, 0),
'umbrella': (0, 255, 0),
'handbag': (0, 255, 0),
'tie': (0, 255, 0),
'suitcase': (0, 255, 0),
'frisbee': (0, 255, 0),
'skis': (0, 255, 0),
'snowboard': (0, 255, 0),
'sports ball': (0, 255, 0),
'kite': (0, 255, 0),
'baseball bat': (0, 255, 0),
'baseball glove': (0, 255, 0),
'skateboard': (0, 255, 0),
'surfboard': (0, 255, 0),
'tennis racket': (0, 255, 0),
'bottle': (0, 255, 0),
'wine glass': (0, 255, 0),
'cup': (0, 255, 0),
'fork': (0, 255, 0),
'knife': (0, 255, 0),
'spoon': (0, 255, 0),
'bowl': (0, 255, 0),
'banana': (0, 255, 0),
'apple': (0, 255, 0),
'sandwich': (0, 255, 0),
'orange': (0, 255, 0),
'broccoli': (0, 255, 0),
'carrot': (0, 255, 0),
'hot dog': (0, 255, 0),
'pizza': (0, 255, 0),
'donut': (0, 255, 0),
'cake': (0, 255, 0),
'chair': (0, 255, 0),
'couch': (0, 255, 0),
'potted plant': (0, 255, 0),
'bed': (0, 255, 0),
'dining table': (0, 255, 0),
'toilet': (0, 255, 0),
'tv': (0, 255, 0),
'laptop': (0, 255, 0),
'mouse': (0, 255, 0),
'remote': (0, 255, 0),
'keyboard': (0, 255, 0),
'cell phone': (0, 255, 0),
'microwave': (0, 255, 0),
'oven': (0, 255, 0),
'toaster': (0, 255, 0),
'sink': (0, 255, 0),
'refrigerator': (0, 255, 0),
'book': (0, 255, 0),
'clock': (0, 255, 0),
'vase': (0, 255, 0),
'scissors': (0, 255, 0),
'teddy bear': (0, 255, 0),
'hair drier': (0, 255, 0),
'toothbrush': (0, 255, 0)
}
def draw_bbox(imgcv, result, conf_thresh=0.6):
    result = result[0]
    for idx, box in enumerate(result['boxes']):
        # print(box)
        x1, y1, x2, y2 = (
            int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        conf = result['scores'][idx]
        # print(conf)
        # print(idx, result['labels'])
        if result['labels'][idx] > len(categories):
            continue
        label = categories[result['labels'][idx]]
        # print(label)
        if label not in category_filter:
            continue
        if conf < conf_thresh:
            continue
        # print(x1,y1,x2,y2,conf,label)
        cv2.rectangle(imgcv, (x1, y1), (x2, y2), cate_to_colors[label], 2)
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)

        if label not in ['person']:
            _x1 = x1
            _y1 = y1  # + int(labelSize[0][1]/2)
            _x2 = _x1 + labelSize[0][0] + 20
            _y2 = y1 - int(labelSize[0][1])
            cv2.rectangle(imgcv, (_x1, _y1), (_x2, _y2), cate_to_colors[label], cv2.FILLED)
            cv2.putText(imgcv, label+str(round(conf.data.tolist(), 2)), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
    return imgcv


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model.to(device)

    postprocessors['bbox'] = postprocessors['bbox'].to(device)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.eval()
    transform = transforms.Compose([
        transforms.Resize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture('/home/lei/Downloads/la_drive.mp4')
    counter = 20000
    cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.namedWindow("test", 2)          
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    ret = True
    while ret:
        # if counter > 6000:
        #     break
        ret, img = cap.read()

        img_canvas = img.copy()
        img = Image.fromarray(img)

        input = transform(img)
        input = input[np.newaxis, ...]
        input = input.to(device, dtype=torch.float32)
        t = time.time()
        output = model(input)
        print((time.time() - t) * 1000, 'ms')
        orig_target_sizes = torch.Tensor([[720, 1280]])
        orig_target_sizes = orig_target_sizes.to(device)

        results = postprocessors['bbox'](output, orig_target_sizes)

        img_canvas = draw_bbox(img_canvas, results, conf_thresh=0.85)

        counter += 1


        cv2.imshow("test", img_canvas)
        key = cv2.waitKey(1)
        if key > 5:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
