''' RenderInterface class provides the method to create a scene from scratch,
Placing the objects in the scene along with their own manipulations '''
import bpy
import os
import sys
import time
import uuid
import json
from BlenderAPI import *

class RenderInterface(object):
    def __init__(self, num_images=None, resolution=300, samples=128):
        self.num_images = num_images # not have used yet
        self.scene = None
        self.setup_blender(resolution, samples)
    
    def setup_blender(self, resolution, samples):
        C = bpy.context
        C.scene.render.engine='CYCLES'
        try:
            C.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            C.user_preferences.addons['cycles'].preferences.devices[0].use = True
        except:
            print("Warning: CUDA device not detected, using CPU instead!", file=sys.stderr)
        self.scene = BlenderScene(bpy.data.scenes[0])
        ## directly giving in the reference of blender object
        # cube = BlenderObject(reference = bpy.data.objects['Cube'])
        # fetching the object in blender using the object name
        cube = BlenderObject(name='Cube')
        cube.set_scale((2,2,2)) #[EXPERIMENTAL]
        cube.delete()
        cam = BlenderCamera(bpy.data.objects['Camera'])
        self.scene.add_camera(cam)
        self.scene.set_render(resolution, samples, set_high_quality=True)


    def render(self, render_path):
        self.scene.render_to_file(render_path)

    def dry_run(self, ):
        ''' Testing building a scene by manual placement'''
        #Json generated from jupyter NB, imports as dataframe
        #index by object name
        #available params are shelves, path, origin, scale_factor
        #Not sure best place to put this line
        with open('src/object_dict.json') as f:
            object_dict = json.load(f)
    
        fridge = self.scene.import_object(filepath='./workspace/objects/fridge_base.dae', \
            overwrite=False, \
            fixed=True)

        apple = self.scene.import_object(filepath='./workspace/objects/apple/manzana2.obj', \
            scale=(0.001368,0.001368,0.001368), \
            location=(0.19139, -0.10539, 1.284), \
            orientation=(-0.778549, -0.057743, 0.137881), \
            fixed=False)
        # apple.set_location(0.19139, -0.10539, 1.28)
        # apple.set_scale((0.001368, 0.001368, 0.001368))
        # apple.set_euler_rotation(-0.778549, -0.057743, 0.137881)

        tomato = self.scene.import_object(filepath='./workspace/objects/tomato/Tomato_v1.obj', \
            scale=(0.009365,0.009365,0.009365), \
            location= (0, -.3, 1.28), \
            orientation= (-.175, 0, 0), \
            fixed=False)
        # tomato.set_scale((0.009365, 0.009365, 0.009365))
        # tomato.set_location(0, -.3, 1.28)
        # tomato.set_euler_rotation(-.175, 0, 0)

        # # NOTE: If object's location, scale, orientation needed to be ketp as in the object file
        # # pass the argument, overwrite=False

        wine = self.scene.import_object(filepath='./workspace/objects/750ML_Wine/750ML_Wine.obj', \
             fixed=False)
        wine.set_scale((0.014387,0.014387,0.014387))
        wine.set_location(0.1, -.32, 1.2245)
        
        # adding light
        bpy.ops.object.light_add(type='POINT', radius=0.25, align='WORLD', location=(-0.25, -1, 1.5))

        # chaning scene camera parameters
        self.scene.camera.set_location(0, -2.5, 2)
        self.scene.camera.set_euler_rotation(1.26929, 0.0138826, -0.120164)

    def place_all(self, ):
        """
        Place all items at origin. Items can then be shuffled for n iterations and rendered for each placement.
        """
        #Json generated from jupyter NB, imports as dataframe
        #index by object name
        #available params are shelves, path, origin, scale_factor
        #Not sure best place to put this line
        with open('src/object_dict.json') as f:
            object_dict = json.load(f)

        #fridge object
        fridge = self.scene.import_object(filepath='./workspace/objects/fridge_base.dae', \
            overwrite=False, \
            fixed=True)

        # adding light
        bpy.ops.object.light_add(type='POINT', radius=0.25, align='WORLD', location=(-0.25, -1, 1.5))

        # changing scene/render conditions camera parameters
        self.scene.camera.set_location(0, -1.75, 1.75)
        self.scene.camera.set_euler_rotation(1.45, 0, 0)

        for key in object_dict.keys():
            sc_fact = object_dict[key]['scale_factor']
            scale = [sc_fact] * 3
            self.scene.import_object(filepath=object_dict[key]['path'], scale=scale)

        #Sometimes items will move without their origin. Trying to clear/reset origin on import
        #Don't want it to apply to fridge though
        #force update first, due to some overwrites on import from original state
        #bpy.context.view_layer.update()
        #for obj in self.scene.objects_unfixed:        
        #    obj.origin_clear()
        #    obj.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

    def shuffle_objects(self, ):
        # random placement of Apple
        with open('src/object_dict.json') as f:
            object_dict = json.load(f)
        for obj in self.scene.objects_unfixed:
            obj.set_location(0, 0, 0)
            obj.place_randomly(object_dict[obj.name])
            print(obj.name)

