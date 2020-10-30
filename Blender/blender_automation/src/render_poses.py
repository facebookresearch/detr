''' Blender Script called from in the pipline to
    Initiate the blender and following the script
    , render images from the given objects. '''

import sys
import os
import bpy
import json
import argparse
from time import time

temp_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(temp_path))

# To import python packages like pandas
#TODO generalize this path
#sys.path.append('/home/solus/anaconda3/lib/python3.8/site-packages')
sys.path.append(r"C:\Users\nickh\anaconda3\Lib\site-packages")
sys.path.append(os.getcwd())
#print(sys.path)
from utils import delete_all
from RenderInterface import RenderInterface

start = time()

# writing file for annotations
write_annotation = open('annotations.csv', 'w')

# Creating a RenderInterface which would be doing all the importing and
# placement of the objects, along with the scene/rendering setup
RI = RenderInterface(num_images=1, resolution=1000)
RI.place_all(repeat_objects=True)

for i in range(500):
    single_img_time = time()
    RI.shuffle_objects()
    shuffle_time = time()

    # Render the scene to a file
    print(f'Starting rendering on image {i}')
    file_path = os.path.abspath(f'./workspace/outputs/test_{i}.jpg')
    RI.render(file_path)
    single_img_time_end = time()
    print(f'Image {i} completed in {single_img_time_end - single_img_time} s. Object shuffling took {shuffle_time - single_img_time} s.')

    #annotation creation
    annotation = RI.scene.get_annotation()
    file_path = os.path.abspath('./workspace/outputs/')
    for ann in annotation:
    	start = annotation[ann]
    	end = annotation[ann]
    	write_annotation.write(f'{file_path},{ann},{start[0]},{start[1]},{end[0]},{end[1]}\n')
    # print('annotation: ', annotation)

end = time()
print(f'\n\n\n:: Total time elapsed in rendering and replacements: {end-start}')

