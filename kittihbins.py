# -----------------------------------------------------------------------------------
# To generate bev heading bins
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

def angle2class(angle):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

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

# Save bev data as json file
output_path = '/srip-vol/datasets/KITTI3D/coco/bev_hbins_%s.json' %(split)
with open(output_path, "w") as outfile:
    json.dump(bev_data, outfile)
