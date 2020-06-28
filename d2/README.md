Detectron2 wrapper for DETR
=======

We provide a Detectron2 wrapper for DETR, thus providing a way to better integrate it in the existing detection ecosystem. It can be used for example to easily leverage datasets or backbones provided in Detectron2.

This wrapper currently supports only box detection, and is intended to be as close as possible to the original implementation, and we checked that it indeed match the results. Some notable facts and caveats:
- The data augmentation matches DETR's original data augmentation. This required patching the RandomCrop augmentation from Detectron2, so you'll need a version from the master branch from June 24th 2020 or more recent.
- To match DETR's original backbone initialization, we use the weights of a ResNet50 trained on imagenet using torchvision. This network uses a different pixel mean and std than most of the backbones available in Detectron2 by default, so extra care must be taken when switching to another one. Note that no other torchvision models are available in Detectron2 as of now, though it may change in the future.
- The gradient clipping mode is "full_model", which is not the default in Detectron2.

# Usage

To install Detectron2, please follow the [official installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

## Evaluating a model

For convenience, we provide a conversion script to convert models trained by the main DETR training loop into the format of this wrapper. To download and convert the main Resnet50 model, simply do:

```
python converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth
```

You can then evaluate it using:
```
python train_net.py --eval-only --config configs/detr_256_6_6_torchvision.yaml  MODEL.WEIGHTS "converted_model.pth"
```


## Training

To train DETR on a single node with 8 gpus, simply use:
```
python train_net.py --config configs/detr_256_6_6_torchvision.yaml --num-gpus 8
```
