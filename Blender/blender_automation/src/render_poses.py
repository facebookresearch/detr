''' Blender Script called from in the pipline to
	Initiate the blender and following the script
	, render images from the given objects. '''

import sys
import os
import bpy
import json
import argparse

sys.path.append('/home/solus/blender_automation/src')
from RenderInterface import RenderInterface
# Creating a RenderInterface which would be doing all the importing and
# placement of the objects, along with the scene/rendering setup
RI = RenderInterface(num_images=1, resolution=1000)

# calling a script manually importing all objects and creating a scene
RI.dry_run() # manuallt placed objects

# finally render the scene to a file
RI.render(os.path.abspath('./../workspace/test.jpg'))