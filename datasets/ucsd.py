"""
For UCSD dataset.

"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

def build(image_set, args, annotation_name=None):
    """
    image_set = 'train' / 'val'
    category = "all/tflt/"
    """
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} to custom dataset does not exist'

    if args.annotation_name is not None:
        training_json_file = 'annotations_train_' + args.annotation_name + '.json'
        validation_json_file = 'annotations_val_' + args.annotation_name + '.json'

    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'annotations_train_{annotation_name}.json'),
        "val": (root / "val2017", root / "annotations" / f'annotations_val_{annotation_name}.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset