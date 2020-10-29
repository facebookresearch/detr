''' Blender Script called from in the pipline to
    Initiate the blender and following the script
    , render images from the given objects. '''

import sys
import os
import bpy
import json
import argparse
from time import time

# To import implemented Blender API, now generalized
#sys.path.append('/home/solus/blender_automation/src')
temp_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(temp_path))

# To import python packages like pandas
#TODO generalize this path
#sys.path.append('/home/solus/anaconda3/lib/python3.8/site-packages')
sys.path.append(r"C:\Users\nickh\anaconda3\Lib\site-packages")
#print(sys.path)

from RenderInterface import RenderInterface

# Creating a RenderInterface which would be doing all the importing and
# placement of the objects, along with the scene/rendering setup
RI = RenderInterface(num_images=1, resolution=1000)
RI.place_all() # manually placed objects

# calling a script manually importing all objects and creating a scene
start = time()

# writing file for annotations
write_annotation = open('annotations.csv', 'w')
for i in range(50):
    RI.shuffle_objects()
    # finally render the scene to a file
    print('Starting rendering...')
    file_path = os.path.abspath(f'./workspace/outputs/test_{i}.jpg')
    RI.render(file_path)
    print(f'Image {i} completed')
    annotation = RI.scene.get_annotation()
    file_path = os.path.abspath('./workspace/outputs/')
    for ann in annotation:
    	start = annotation[ann]
    	end = annotation[ann]
    	write_annotation.write(f'{file_path},{ann},{start[0]},{start[1]},{end[0]},{end[1]}\n')
    # print('annotation: ', annotation)
end = time()

print(f'\n\n\n:: Total time elapsed in rendering and replacements: {end-start}')

