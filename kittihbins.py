# -----------------------------------------------------------------------------------
# To generate bev bins
# -----------------------------------------------------------------------------------
import os 
import os.path as osp
import json
# import torch
from tqdm import tqdm
import numpy as np
import cv2

def ry2alpha(ry, u, cu, fu):
    alpha = ry - np.arctan2(u - cu, fu)

    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi

    return alpha

def angle2class(angle):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(12)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

split = 'val'
data_path = '/srip-vol/datasets/KITTI3D/coco/kitti_%s.json' %(split)
data = json.load(open(data_path))
heading_bins_data = {}
heading_ress_data = {}

KITTI_CLASS = {'Car': 0, 'Pedestrian': 1, 'Cyclist' : 2}

for i in tqdm(range(len(data['images']))):
    # Label path
    img_path = data['images'][i]['file_name'].split('/')
    img_path[-2] = 'label_2'
    img_path[-1] = img_path[-1].split('.')[0] + '.txt'
    label_path = '/' + osp.join(*img_path)
    
    # Calibration path
    img_path = data['images'][i]['file_name'].split('/')
    img_path[-2] = 'calib'
    img_path[-1] = img_path[-1].split('.')[0] + '.txt'
    calib_path = '/' + osp.join(*img_path)

    # Read annotations and assembler point depth value
    lines = open(label_path).readlines()
    lines_calib = open(calib_path).readlines()

    obj = lines_calib[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    # obj = lines_calib[3].strip().split(' ')[1:]
    # P3 = np.array(obj, dtype=np.float32)
    # obj = lines_calib[4].strip().split(' ')[1:]
    # R0 = np.array(obj, dtype=np.float32)
    # obj = lines_calib[5].strip().split(' ')[1:]
    # Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    P2 = P2.reshape(3, 4)
    # P3.reshape(3, 4)
    # R0.reshape(3, 3)
    # Tr_velo_to_cam.reshape(3, 4)

    # cv = P2[1, 2]
    # fv = P2[1, 1]
    # tx = P2[0, 3] / (-fu)
    # ty = P2[1, 3] / (-fv)
    cu = P2[0, 2]
    fu = P2[0, 0]
    
    heading_bins = []
    heading_ress = []
    for line in lines:
        label_data = line.split(' ')
        if(KITTI_CLASS.get(label_data[0],-1) == -1):
            continue
        bbx0 = float(label_data[4])
        bbx2 = float(label_data[6])
        ry = float(label_data[14])

        heading_angle = ry2alpha(ry, (bbx0 + bbx2) / 2, cu, fu)
        if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
        if heading_angle < -np.pi: heading_angle += 2 * np.pi
        heading_bin, heading_res = angle2class(heading_angle)
        # x_c = float(label_data[11])
        # z_c = float(label_data[13])

        heading_bins.append(heading_bin)
        heading_ress.append(heading_res)

    # Save in dict
    heading_bins_data[i+1] = heading_bins #torch.tensor(depth) 
    heading_ress_data[i+1] = heading_ress

# Save bev data as json file
output_path = '/srip-vol/datasets/KITTI3D/coco/heading_bins_%s.json' %(split)
with open(output_path, "w") as outfile:
    json.dump(heading_bins_data, outfile)

output_path = '/srip-vol/datasets/KITTI3D/coco/heading_ress_%s.json' %(split)
with open(output_path, "w") as outfile:
    json.dump(heading_ress_data, outfile)
