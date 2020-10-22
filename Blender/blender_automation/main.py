""" blender_automation.py 
Description : This file takes input of all the model files and executable
files necessary for the compelete module. Also the parameters like number of
images to be generated and other parameters like that. """

import os
import sys
import argparse
from src.rendering import render_pipeline

def parser():
    parser.ArgumentParser(description='Calls necessary module for rendering \
            and saving images in the output directory following the passed \
            parameters')
    parser.add_argument('--BLENDER-PATH', default='/usr/bin/blender', \
            help='blender executable file path')
    parser.add_argument('--OBJECTS-DIR', default='./workspace/objects', \
            help='CAD Objects files path')
    parser.add_argument('--OUTPUT-DIR', default='./workspace/outputs', \
            help='Output path for storing the renderd image')
    parser.add_argument('--N', default=10, help='Total number of images')
    parser.add_argument('--DIM', default=300, help='Dimension for rendering images')
    args = parser.parse_args()

def main(args):
    ############################### INPUT PATHS ###############################
    # path to render workspace folder
    workspace = os.path.join(os.getcwd(),"workspace")
    # path to folder containing a set of .model files
    obj_set = os.path.join(workspace, 'object_files') # obj files

    ################################ RENDERING ################################
    # Rendering Parameters
    blender_attributes = {
        "attribute_distribution_params":
            [
                # number of lamps is a DISCRETE UNIFORM DISTRIBUTION over NON_NEGATIVE INTEGERS,
                # params l and r are lower and upper bounds of distributions, need to be positive integers
                ["num_lamps","mid", 6], ["num_lamps","scale", 0.3],

                # lamp energy is a TRUNCATED NORMAL DISTRIBUTION, param descriptions same as above
                ["lamp_energy", "mu", 5000.0], ["lamp_energy", "sigmu", 0.3],

                # camera location is a COMPOSITE SHELL RING DISTRIBUTION
                # param normals define which rings to use, based on their normals, permitted values are 'X','Y','Z' and a combination of the three
                # phi sigma needs to be non-negative, and defines the spread of the ring in terms of degrees
                # phi sigma of roughly 30.0 corresponds to a unifrom sphere
                ["camera_loc","phi_sigma", 10.0],

                # camera radius is a Truncated Normal Distribution
                ["camera_radius", "mu", 6.0], ["camera_radius", "sigmu", 0.3],
            ],
        "attribute_distribution" : []
    }

    ############################################################################

    ############################################################################
    ################################ EXECUTION ################################
    ############################################################################

    # construct rendering parameters
    arguments = {
        "objects_dir": args.OBJECTS_DIR,
        "blender_path": args.BLENDER_PATH,
        "work_dir": workspace,
        "blender_attributes": blender_attributes
        }

    # run blender pipeline and produce a zip with all rendered images
    path_of_zip = render_pipeline.full_run(**arguments)
    

if __name__ == '__main__':
    args = parser()
    main(parser)
