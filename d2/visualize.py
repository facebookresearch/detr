# from https://gist.github.com/JosephKJ/e523f77af6c5768538eda0c7e2af375a
import os

import cv2
from detectron2.utils.logger import setup_logger

from d2.detr import add_detr_config

setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

BASE = '/home/benedikt/PycharmProjects/pytorch-deeplab-xception/data/cityscapes/leftImg8bit/train'

imgs = [
    'hamburg/hamburg_000000_027304_leftImg8bit.png',
    'hamburg/hamburg_000000_032906_leftImg8bit.png',
    'zurich/zurich_000067_000019_leftImg8bit.png',
    'ulm/ulm_000014_000019_leftImg8bit.png',
    'ulm/ulm_000019_000019_leftImg8bit.png'
]

imgs = list(map(lambda x: os.path.join(BASE, x), imgs))


# Get image
im = cv2.imread(imgs[0])

# Get the configuration ready
cfg = get_cfg()
add_detr_config(cfg)
cfg.merge_from_file("configs/detr_citypersons_256_6_6_torchvision.yaml")
cfg.MODEL.WEIGHTS = "output/model_0001399.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img = v.get_image()[:, :, ::-1]
cv2.imwrite('output.jpg', img)