# -----------------------------------------------------------------------------------
# To generate bev data
# -----------------------------------------------------------------------------------
import os 
import os.path as osp
import json
# import torch
from tqdm import tqdm

split = 'val'
data_path = '/srip-vol/datasets/KITTI3D/coco/kitti_%s.json' %(split)
data = json.load(open(data_path))
bev_data = {}

KITTI_CLASS = {'Car': 0, 'Pedestrian': 1, 'Cyclist' : 2}

for i in tqdm(range(len(data['images']))):
    # Label path
    img_path = data['images'][i]['file_name'].split('/')
    img_path[-2] = 'label_2'
    img_path[-1] = img_path[-1].split('.')[0] + '.txt'
    label_path = '/' + osp.join(*img_path)

    # Read annotations and assembler point depth value
    lines = open(label_path).readlines()
    bev = []
    for line in lines:
        label_data = line.split(' ')
        if(KITTI_CLASS.get(label_data[0],-1) == -1):
            continue
        x_c = float(label_data[11])
        z_c = float(label_data[13])

        bev.append([x_c, z_c])

    # Save in dict
    bev_data[i+1] = bev #torch.tensor(depth) 
