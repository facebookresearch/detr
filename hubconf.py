# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from models.backbone import Backbone, Joiner
from models.detr import DETR, PostProcess
from models.position_encoding import PositionEmbeddingSine
from models.segmentation import DETRsegm, PostProcessPanoptic
from models.transformer import Transformer

dependencies = ["torch", "torchvision"]


def _make_detr(backbone_name: str, dilation=False, num_classes=91, mask=False):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    detr = DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=100)
    if mask:
        return DETRsegm(detr)
    return detr


def detr_resnet50(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    DETR R50 with 6 encoder and 6 decoder layers.

    Achieves 42/62.4 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet50", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def detr_resnet50_dc5(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    DETR-DC5 R50 with 6 encoder and 6 decoder layers.

    The last block of ResNet-50 has dilation to increase
    output resolution.
    Achieves 43.3/63.1 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet50", dilation=True, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def detr_resnet101(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    DETR-DC5 R101 with 6 encoder and 6 decoder layers.

    Achieves 43.5/63.8 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet101", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def detr_resnet101_dc5(pretrained=False, num_classes=91, return_postprocessor=False):
    """
    DETR-DC5 R101 with 6 encoder and 6 decoder layers.

    The last block of ResNet-101 has dilation to increase
    output resolution.
    Achieves 44.9/64.7 AP/AP50 on COCO val5k.
    """
    model = _make_detr("resnet101", dilation=True, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def detr_resnet50_panoptic(
    pretrained=False, num_classes=250, threshold=0.85, return_postprocessor=False
):
    """
    DETR R50 with 6 encoder and 6 decoder layers.
    Achieves 43.4 PQ on COCO val5k.

   threshold is the minimum confidence required for keeping segments in the prediction
    """
    model = _make_detr("resnet50", dilation=False, num_classes=num_classes, mask=True)
    is_thing_map = {i: i <= 90 for i in range(250)}
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessPanoptic(is_thing_map, threshold=threshold)
    return model


def detr_resnet50_dc5_panoptic(
    pretrained=False, num_classes=91, threshold=0.85, return_postprocessor=False
):
    """
    DETR-DC5 R50 with 6 encoder and 6 decoder layers.

    The last block of ResNet-50 has dilation to increase
    output resolution.
    Achieves 44.6 on COCO val5k.

   threshold is the minimum confidence required for keeping segments in the prediction
    """
    model = _make_detr("resnet50", dilation=True, num_classes=num_classes, mask=True)
    is_thing_map = {i: i <= 90 for i in range(250)}
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessPanoptic(is_thing_map, threshold=threshold)
    return model


def detr_resnet101_panoptic(
    pretrained=False, num_classes=91, threshold=0.85, return_postprocessor=False
):
    """
    DETR-DC5 R101 with 6 encoder and 6 decoder layers.

    Achieves 45.1 PQ on COCO val5k.

   threshold is the minimum confidence required for keeping segments in the prediction
    """
    model = _make_detr("resnet101", dilation=False, num_classes=num_classes, mask=True)
    is_thing_map = {i: i <= 90 for i in range(250)}
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcessPanoptic(is_thing_map, threshold=threshold)
    return model
