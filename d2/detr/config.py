# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_detr_config(cfg):
    """
    Add config for DETR.
    """
    cfg.MODEL.DETR = CN()
    cfg.MODEL.DETR.NUM_CLASSES = 80

    # For Segmentation
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ''

    # LOSS
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.DETR.DEEP_SUPERVISION = True
    cfg.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1

    # TRANSFORMER
    cfg.MODEL.DETR.NHEADS = 8
    cfg.MODEL.DETR.DROPOUT = 0.1
    cfg.MODEL.DETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DETR.ENC_LAYERS = 6
    cfg.MODEL.DETR.DEC_LAYERS = 6
    cfg.MODEL.DETR.PRE_NORM = False

    cfg.MODEL.DETR.HIDDEN_DIM = 256
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
