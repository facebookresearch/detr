''' RenderInterface class provides the necessary high level  methods to create
    a scene from scratch in Blender, Placing the objects in the scene along with
    their own manipulations '''
import bpy
import os
import sys
import time
import uuid
import json
from BlenderAPI import *

class RenderInterface(object):
    ''' RenderInterface Class represents the command line toolkit like modules that
        can be used in writing scripts to create a scene, and automate its manipulation
        Attributes:
            scene: BlenderScene Class Object
                BlenderScene Class object which can manage all the objects and the scene rendering configuration
        Methods:
            __init__(resolution=300, samples=128)
                Constructor for the Class RenderInterface that initialize the Object, set up Blender with class method `setip_blender()`
            setup_blender(resolution, sample)
                prepare the blender scene to be edited, removes default Cube object, add camera to the BlenderScene object, set render configurations
            render(render_path)
                Calls BlenderScene Class method `render_to_file` to render the scene and save the generated image to a file
            dry_run()
                Complete manual scripts for testing modules, no part in automation
            place_all(repeat_objects=False)
                It's more like Import All objects, at the origin of the 3d space according to a parameters passed as a JSON file, refer to the method's documentation for info on how to create JSON file for parameters
            shuffle_objects()
                Shuffle Objects in the Scene with the constraints passed as a JSON file, Look at the method documentation for information about parameters passed as JSON file
    '''
    def __init__(self, resolution=300, samples=128):
        ''' RenderInterface Class Object constructor, initializes scene, and setup rendering configuration
            Parameters:
                resolution: int, optional
                    Resolution for the rendered image need to be configured in Blender Rendering settings
                    Default is 300
                samples: int, optional
                    Number of samples to be rendered, Default is 128. NOTE: Not being used as of now
        '''
        self.scene = None
        self.setup_blender(resolution, samples)
    
    def setup_blender(self, resolution, samples):
        ''' Method to setup the parameters in rendering configuration in Blender, called once in the constructor
            Clears up the default cube in the blender scene when initialized for the first time, add camera to
            Blender Scene object/Attribute of RenderInterface class object.
            Parameters:
                resolution: int
                    resolution to be set in rendering setting for image creation of the scene
                sample: int
                    Number of samples to be generated in the scene
        '''
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
        cube.delete()
        cam = BlenderCamera(bpy.data.objects['Camera'])
        self.scene.add_camera(cam)
        self.scene.set_render(resolution, samples, set_high_quality=True)


    def render(self, render_path):
        ''' Class Method to render the Blender Scene into an image and write it in the file with the given path
            Parameters:
                render_path: string
                    render image into the given parameter `render_path`, by calling BlenderScene Class method `render_to_file`
        '''
        self.scene.render_to_file(render_path)

    def dry_run(self, ):
        ''' Testing building a scene by manual placement
        '''
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

    def place_all(self, repeat_objects=False):
        """
        Place all items at origin. Items can then be shuffled for n iterations and rendered for each placement.
        Parameters:
            repeat_objects: Bool
                if True, Objects will be repeated according to the parameters set in JSON file
                otherwise, each object will be spawned only once in the Blender Scene
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
        
        if repeat_objects:
            for key in object_dict.keys():
                repeats = random.randint(1,object_dict[key]['max_repeats'])
                print(f'{key} will be printed {repeats} times')
                for _ in range(repeats):
                    sc_fact = object_dict[key]['scale_factor']
                    scale = [sc_fact] * 3
                    self.scene.import_object(filepath=object_dict[key]['path'], scale=scale, orientation=object_dict[key]['import_rotations'])
        else:
            for key in object_dict.keys():
                sc_fact = object_dict[key]['scale_factor']
                scale = [sc_fact] * 3
                self.scene.import_object(filepath=object_dict[key]['path'], scale=scale, orientation=object_dict[key]['import_rotations'])


    def shuffle_objects(self, ):
        ''' Class Method to shuffle all the objects keeping them in the constraint mentioned in JSON file containing object parameters.
        '''
        with open('src/object_dict.json') as f:
            object_dict = json.load(f)
        for obj in self.scene.objects_unfixed:
            obj.set_location(0, 0, 0)
            #bad hack for multiple object placement, will break for max_repeat>9
            if '.00' in obj.name:
                obj_name = obj.name[:-4]
            else:
                obj_name = obj.name
            obj.place_randomly(object_dict[obj_name])
            print(obj.name)

