# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Helper script to convert models trained with the main version of DETR to be used with the Detectron2 version.
"""
import json
import argparse

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("D2 model converter")

    parser.add_argument("--source_model", default="", type=str, help="Path or url to the DETR model to convert")
    parser.add_argument("--output_model", default="", type=str, help="Path where to save the converted model")
    return parser.parse_args()


def main():
    args = parse_args()

    # D2 expects contiguous classes, so we need to remap the 92 classes from DETR
    # fmt: off
    coco_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91]
    # fmt: on

    coco_idx = np.array(coco_idx)

    if args.source_model.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(args.source_model, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(args.source_model, map_location="cpu")
    model_to_convert = checkpoint["model"]

    model_converted = {}
    for k in model_to_convert.keys():
        old_k = k
        if "backbone" in k:
            k = k.replace("backbone.0.body.", "")
            if "layer" not in k:
                k = "stem." + k
            for t in [1, 2, 3, 4]:
                k = k.replace(f"layer{t}", f"res{t + 1}")
            for t in [1, 2, 3]:
                k = k.replace(f"bn{t}", f"conv{t}.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            k = "backbone.0.backbone." + k
        k = "detr." + k
        print(old_k, "->", k)
        if "class_embed" in old_k:
            v = model_to_convert[old_k].detach()
            if v.shape[0] == 92:
                shape_old = v.shape
                model_converted[k] = v[coco_idx]
                print("Head conversion: changing shape from {} to {}".format(shape_old, model_converted[k].shape))
                continue
        model_converted[k] = model_to_convert[old_k].detach()

    model_to_save = {"model": model_converted}
    torch.save(model_to_save, args.output_model)


if __name__ == "__main__":
    main()
