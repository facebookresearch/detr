import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO

def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    x_c, y_c, w, h = out_bbox.unbind(1)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs, im, threshold):

    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    probas_to_keep = probas[keep]
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas_to_keep, bboxes_scaled

def generate_conf_outputs(my_image, my_model, threshold):
    """
    Takes an input image, and outputs predictions formatted for confusion
    matrix functions. 
    Arguments: 
        my_image - PIL instance of an image
        my_model - model used for inference
        threshhold - confidence above which boxes should be used. Default 0.5
    Returns:
        BBox predictions for image formatted for use on matrix generation
    """
  # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)
  # propagate through the model
    outputs = my_model(img)
    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,my_image, 
                                                             threshold)
    class_pred = []
    probability = []
    x1, y1, x2, y2 = ([] for i in range(4))
    if probas_to_keep is not None and bboxes_scaled is not None:
        for p, (xmin,ymin,xmax,ymax) in zip(probas_to_keep, bboxes_scaled.tolist()):
            cl_ind = p.argmax()
            class_pred.append(cl_ind.item())
            probability.append(p[cl_ind].item())
            x1.append(xmin)
            y1.append(ymin)
            x2.append(xmax)
            y2.append(ymax)

    formatted_outputs = list(zip(x1, y1, x2, y2, probability, class_pred))
    return np.asarray(formatted_outputs)

def getImg(image_name, images):
  for i in images:
    if i["file_name"] == image_name:
      return i['id']

def getActualBBoxes(image_id, annotations):
  listOfBoxes = []
  for annot in annotations:
    if annot["image_id"] == image_id:
      bbox_wh= list(annot["bbox"])
      bbox_conv = (bbox_wh[0], bbox_wh[1],
                   bbox_wh[0]+bbox_wh[2], bbox_wh[1]+bbox_wh[3])
      cat_index = annot["category_id"]
      listOfBoxes.append([cat_index] + list(bbox_conv))
  return np.asarray(listOfBoxes)

def run_validation_batch(conf_mat, validation_images, coco_anno_path, model, list_classes, threshold=0.2):
    """
    Function to run confusion matrix validation on all images in validation path. 
    Arguments:
        conf_mat = confusion matrix object
        validation_images = iterable of all images in validation folder (Created using glob)
        coco_anno_path = path to coco formatted validation ground truth json
        model = model to be used for inference
        list_classes = list of class names for labeling heat map
        threshold = confidence threshold for predictions
    Returns: 
        confusion matrix np array. 
    """

    coco=COCO(coco_anno_path)
    annotations = coco.loadAnns(coco.getAnnIds())
    image_list = coco.loadImgs(coco.getImgIds())

    for imagepath in validation_images:
        img_name = Path(imagepath).parts[-1]
        im = Image.open(imagepath)
        formatted_preds = generate_conf_outputs(im, model, threshold)
        img_id = getImg(img_name, image_list)
        gt_boxes = getActualBBoxes(img_id, annotations)
        conf_mat.process_batch(formatted_preds, gt_boxes)

        
    conf_result_df = pd.DataFrame(conf_mat.return_matrix(), columns=list_classes, index=list_classes)
    f, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(conf_result_df, annot=True,  fmt='.0f')
    ax.set(title='Confusion Matrix', xlabel='predicted', ylabel='Actual')
    return plt.show()


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''

        if np.size(detections) != 0: #If size detections>0, filter to those above conf. threshold
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        else: #create null set of detections
            detections = np.empty((1,6))

        gt_classes = labels[:, 0].astype(np.int16)
        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)
        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
        
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0: # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index = True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index = True)[1]]

        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[(gt_class), detection_class] += 1
            else:
                gt_class = gt_classes[i]
                self.matrix[(gt_class), 0] += 1
        
        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[0 ,detection_class] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))
        
