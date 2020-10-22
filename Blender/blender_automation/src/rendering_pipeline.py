''' 
##########################################################################################
############    FOR LATER USE: Would execute from start to end the whole rendering process
##########################################################################################
import os
import sys

def generate_poses(src_dir, blender_path, object_dir, output_dir, render_per_product, blender_attributes, visualize_dump=False, dry_run_mode=False, render_resolution=300, render_sample=128):
    print(f'src directory: {src_dir}\nWill be added the blender running environment')
    print(f'blender\'s path: {blender_path}')
    blender_script_path = os.path.join(os.path.abspath(src_dir), 'rendering', 'render_poses.py')
    blender_args = [blender_path, '--background', '--python', blender_script_path,\
            src_dir, \
            objects_dir, \
            output_dir, \
            str(render_per_product), \
            str(render_resolution), \
            str(render_sample), \
            ]

def full_run(objects_dir, blender_path, work_dir=workspace, blender_attribute={}, \
        n_of_pixels = 300, render_samples=128, render_per_class=10):
    for object_dir in os.listdir(os.path.abspath(objects_dir)):

'''