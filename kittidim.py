# -----------------------------------------------------------------------------------
# To generate bev dimension data
# -----------------------------------------------------------------------------------
import os 
import os.path as osp
import json
# import torch
from tqdm import tqdm

split = 'val'
data_path = '/srip-vol/datasets/KITTI3D/coco/kitti_%s.json' %(split)
data = json.load(open(data_path))
bevdim_data = {}

KITTI_CLASS = {'Car': 0, 'Pedestrian': 1, 'Cyclist' : 2}

for i in tqdm(range(len(data['images']))):
    # Label path
    img_path = data['images'][i]['file_name'].split('/')
    img_path[-2] = 'label_2'
    img_path[-1] = img_path[-1].split('.')[0] + '.txt'
    label_path = '/' + osp.join(*img_path)

    # Read annotations and assembler point depth value
    lines = open(label_path).readlines()
    bevdim = []
    for line in lines:
        label_data = line.split(' ')
        if(KITTI_CLASS.get(label_data[0],-1) == -1):
            continue
        dim_w = float(label_data[9])
        dim_l = float(label_data[10])

        bevdim.append([dim_w, dim_l])

    # Save in dict
    bevdim_data[i+1] = bevdim #torch.tensor(depth) 

# Save bev data as json file
output_path = '/srip-vol/datasets/KITTI3D/coco/bev_dim_%s.json' %(split)
with open(output_path, "w") as outfile:
    json.dump(bevdim_data, outfile)
