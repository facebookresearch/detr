# -*- coding: utf-8 -*-

# !pip install -q git+https://github.com/huggingface/transformers.git

# !pip install -q timm

"""## Prepare the image using DetrFeatureExtractor

Let's use the image of the two cats chilling on a couch once more. It's part of the [COCO](https://cocodataset.org/#home) object detection validation 2017 dataset.
"""

from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt


"""Let's first apply the regular image preprocessing using `DetrFeatureExtractor`. The feature extractor will resize the image (minimum size = 800, max size = 1333), and normalize it across the channels using the ImageNet mean and standard deviation."""

from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, prob, boxes):
    global model
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("last_detected.jpg")

def detect_image(im):
    global model
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    encoding = feature_extractor(im, return_tensors="pt")
    encoding.keys()

    print(encoding['pixel_values'].shape)

    """## Forward pass

    Next, let's send the pixel values and pixel mask through the model. We use the one with a ResNet-50 backbone here (it obtains a box AP of 42.0 on COCO validation 2017).
    """

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    outputs = model(**encoding)

    """Let's visualize the results!"""

    # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # rescale bounding boxes
    target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    plot_results(im, probas[keep], bboxes_scaled)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    SAMPLE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    parser = argparse.ArgumentParser(description="DETR detection")
    group = parser.add_argument_group('input_type')
    group.add_argument("--path", help="path to images or video")
    group.add_argument("--url", help="URL to image")

    args = parser.parse_args()
    if args.url:
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        im = Image.open(requests.get(url, stream=True).raw)
    elif args.path:
        path = Path(args.path)
        im = Image.open(str(path))

    detect_image(im)
