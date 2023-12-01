import torch
import glob
import sys

import time
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T


from models.backbone import Backbone, Joiner
from models.detr import DETR, PostProcess
from models.position_encoding import PositionEmbeddingSine
from models.segmentation import DETRsegm, PostProcessPanoptic
from models.transformer import Transformer

import hubconf

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


def detr_custom(pretrained=False, num_classes=91, return_postprocessor=False):
    model = _make_detr("resnet50", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/nhphucqt/detr_person/resolve/main/checkpoint_003.pth?download=true", map_location="cpu", check_hash=True
            # url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location="cpu", check_hash=True
            # url="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    if (img.shape[-2] > 1600 or img.shape[-1] > 1600):
        return None, None
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def plot_results(pil_img, prob, boxes):
    print("prob:", prob)
    print("boxes:", boxes)
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def detect_img(img_path, model, transform):
    url = img_path
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    print("Image:", im.size)

    start = time.time()
    scores, boxes = detect(im, model, transform)
    stop = time.time()

    if (scores is None):
        print("Image size too large")
        return

    print(f"Time: {stop - start}s")
    plot_results(im, scores, boxes)

def detect_set(model, transform, path_name):
    dir_path = path_name

    img_set = glob.glob(dir_path + "*.jpg")
    img_set.sort()

    sum_time = 0
    cnt = 0

    for img_path in img_set:
        print(img_path, ":", end=" ")
        im = Image.open(img_path).convert("RGB")
        start = time.time()
        scores, boxes = detect(im, model, transform)
        stop = time.time()
        im.close()

        if (scores is None):
            print("Image size too large")
            continue

        print(len(boxes), ", Time:", stop - start, "s")
        sum_time += stop - start
        cnt += 1
        print("Average time", cnt, ":", sum_time / cnt, "s")
    # mean-std normalize the input image (batch-size: 1)

if __name__ == "__main__":
    # COCO classes
    # CLASSES = [
    #    'person'
    # ]
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detr = detr_custom(pretrained=True, num_classes=1, return_postprocessor=False).eval()
    # detr = hubconf.detr_resnet101_dc5(pretrained=True).eval()

    # url = 'http://images.cocodataset.org/train2017/000000000536.jpg'
    # im = Image.open(requests.get(url, stream=True).raw)

    # print("Image:", im.size)

    # start = time.time()
    # scores, boxes = detect(im, detr, transform)
    # stop = time.time()

    # print(f"Time: {stop - start}s")
    # plot_results(im, scores, boxes)
    path_name = sys.argv[1]
    detect_set(detr, transform, path_name)
    # print("Detected:", detected)

    # detect_img("http://images.cocodataset.org/val2017/000000002299.jpg", detr, transform)