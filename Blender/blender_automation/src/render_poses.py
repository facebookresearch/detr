''' Main Script that will execute in Blender Python environment
    Taking levarage of all the assets, of automatic placement of
    the objects and rendering synthetic data to the image files
    along with the object annotation for the object detection.
'''

import sys
import os
import bpy
import json
import argparse
from time import time

# adding module directory in the Blender Python Environmental PATH Variable
MODULE_DIRECTORY = os.path.realpath(__file__)
sys.path.append(os.path.dirname(MODULE_DIRECTORY))

# To import python packages like pandas
#TODO generalize this path
try: # bad hack just for the time being so that 
    #we don't have to change it everytime we ran on each of our system
    sys.path.append('/home/solus/anaconda3/lib/python3.8/site-packages')
except:
    sys.path.append(r"C:\Users\nickh\anaconda3\Lib\site-packages")

## TODO: Don't need anymore, intended for the import of utils.py which we aren't using anymore
# sys.path.append(os.getcwd())

#TODO : [utils.delete_all] not being used, anyway it's written in BlenderScene Class as Class Method
#from utils import delete_all

from RenderInterface import RenderInterface

start = time()

# Open a file for writing object annotations
write_annotation = open('annotations.csv', 'w')
write_annotation.write('file_path\tobject_name\ttop_left_xy\tright_bottom_xy\n')

# Creating a RenderInterface which would be doing all the importing and
# placement of the objects, along with the scene/rendering setup
RI = RenderInterface(resolution=(400,600), samples=128, set_high_quality=True)


for i in range(1):
  RI.scene.delete_all()
  RI.setup_blender(resolution=(400,600), samples=128, set_high_quality=True)
  RI.place_all(repeat_objects=True)

  imgs_per_shuffle = 1
  for j in range(imgs_per_shuffle):
    single_img_time = time()
    RI.shuffle_objects()

    index = j + imgs_per_shuffle*i
    # Render the scene to a file
    print(f'Starting rendering on image {index}')
    file_path = os.path.abspath(f'./workspace/outputs/image_{index}.jpg')
    RI.render(file_path)
    single_img_time_end = time()
    print(f'Image {index} completed in {single_img_time_end - single_img_time} s.')

    # Fetch annotation from rendered image
    annotation = RI.scene.get_annotation()
    #TODO could be written directly in COCO format
    for ann in annotation:
        object_name = annotation[ann][0]
        left_top = annotation[ann][1]
        right_bottom = annotation[ann][2]
      #TODO import params JSON file and convert 3d model object name to the actual label, Could be part of post annotation processing seperately
        write_annotation.write(f'{file_path}\t{object_name}\t({left_top[0]},{left_top[1]})\t({right_bottom[0]},{right_bottom[1]})\n')

end = time()
print(f'\n\n\n:: Total time elapsed in rendering and replacements: {end-start}')

