from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import os.path as osp
from math import ceil
import os
from tqdm import tqdm

KITTI_CLASS = {'Car': 0, 'Pedestrian': 1, 'Cyclist' : 2}

def decode(label):
     '''
     Parse line of kitti label text file
     Refer - https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#kittidetectiondataset
     '''
     data = label.split(' ')
     class_name = data[0]
     xmin = float(data[4])
     ymin = float(data[5])
     xmax = float(data[6])
     ymax = float(data[7])

     # top left cornet and dimensions
     # Refer - https://cocodataset.org/#format-data
     bbox = [xmin, ymin, ceil(xmax-xmin), ceil(ymax-ymin)]
     class_id = KITTI_CLASS.get(class_name, -1)

     return class_id, class_name, bbox


# Coco object
coco = Coco()

# Add categories
coco.add_category(CocoCategory(id=0, name='Car'))
coco.add_category(CocoCategory(id=1, name='Pedastrian'))
coco.add_category(CocoCategory(id=2, name='Cyclist'))

split = 'val'
assert split in ['train', 'val']

# Add paths
imageset_path = osp.join('/srip-vol/datasets/KITTI3D/ImageSets', split + '.txt')
img_folder_path = osp.join('/srip-vol/datasets/KITTI3D/training', 'image_2')
ann_folder_path = osp.join('/srip-vol/datasets/KITTI3D/training', 'label_2')

idx = open(imageset_path, 'r').readlines()

for i in tqdm(idx):
     i = i[:-1]
     img_path = osp.join(img_folder_path, i +'.png')
     lab_path = osp.join(ann_folder_path, i +'.txt')

     width, height = Image.open(img_path).size
     coco_image = CocoImage(file_name=img_path, height=height, width=width)

     labels = open(lab_path, 'r').readlines()
     for l in labels:
          category_id, category_name, bbox = decode(l)
          if category_id == -1:
               continue
          coco_image.add_annotation(CocoAnnotation(   
               bbox=bbox,
               category_id=category_id,    
               category_name=category_name))

     coco.add_image(coco_image)

save_path = '/srip-vol/parth/detr/kitti_%s.json'%(split)
# save_path = '/srip-vol/parth/detr/try.json'
save_json(data=coco.json, save_path=save_path) 