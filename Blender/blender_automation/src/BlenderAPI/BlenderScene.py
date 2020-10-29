import bpy
import bpy_extras

# relative import from other classes
from .BlenderObject import *

class BlenderScene(object):
    ''' BlenderScene object represents the Scene object of blender, encapsulates methods like spawning object, maintaing them in teh scene, along with the rendering parameters and finally rendering process
        Attributes:
            lambs: list
                list containing lambs in the scene
            background: None #TODO deprecated
            object_fixed: list
                list containing the objects which are not intended to place randomly to create differnt scene
            object_unfixed: list
                list containing the objects which are place randomly in each iteration to create different scene in blender
            subjects: #TOTO deprecated
            subjects_bot: #TODO deprecated
            reference : blender reference object
                blender object used for reference of scene in context / bpy.context.scene

        Methods:
            __init__(reference):
                initialize the class object with its appropriate paramters with taking blender refernece object as argument
            set_render(resolution=300, sample=128, set_high_quality=False):
                set the rendering attributes for the redering process in blender like resolution, engine type, cycles count
            add_object_fixed(obj):
                add BlenderObject class object into the list of the fixed object maintained by the BlenderScene class object
            add_object_unfixed(obj):
                add BlenderObject class object into the list of the unfixed object maintened and used by BlenderScene object to create different geometrical configurations of object in the scene
            import_object(filepath, location=(0,0,0), scale=(1,1,1), orientation=(0,0,0), overwrite=True, fixed=False):
                import object from the 3d model file and create a BlenderObject class object for that, and accordingly add object in the list of fixed/unfixed object list maintained by BlendeScene object
            delete_all():
                delete everything in the scene in blender and in this class object
            remove_object(obj):
                remove object from the scene
            render_to_file(filepath):
                render 2d image of the 3d scene with the perspective of the scene camera
            point_conversion(x, y, z):
                convert point in 3d space in the 2d space projected to the scene camera
            get_annotations():
                generate annotation for all the objects in the list object_unfixed in the format filepath, obj_name, top_left_corner, right_bottom_corner
    '''
    def __init__(self, reference):
        ''' Constructor, initilizes BlenderScene object with it's appropriate parameters and attributes
            Parameters:
                reference: bpy data reference / bpy.context.scene
        '''
        self.lambs = []
        self.background = None #TODO Deprecate not in use
        self.objects_fixed = [] 
        self.objects_unfixed = []
        self.camera = None
        # self.subjects = []
        # self.subjects_bot = []
        self.reference = reference #bpy.data.scenes[0] or bpy.context.scene

    def set_render(self, resolution=300, samples=128, set_high_quality=False):
        ''' Setting up the scene rendering configuration 
            Parameters:
                resolution: int
                    resolution of the image, default is 300. Can be changed at later stage while actually executing rendering command
                sample: int
                    NOT YET USED, intended for the number of images to be generated for the single subject
                set_high_quality: bool, optional
                    if True, render configuration sets to a bit higher quality like more resolution and all.
                    Default is False
        '''
        self.reference.cycles.film_transparent=True
        self.reference.cycles.max_bounces = 1
        self.reference.cycles.max_bounces = 1
        self.reference.cycles.transparent_max_bounces = 1
        self.reference.cycles.transparent_min_bounces = 1
        self.reference.cycles.samples = samples
        self.reference.cycles.device='GPU'
        self.reference.render.tile_x = 512
        self.reference.render.tile_y = 512
        self.reference.render.resolution_x = resolution
        self.reference.render.resolution_y = resolution
        self.reference.render.resolution_percentage = 100
        if set_high_quality:
            self.reference.cycles.samples = 512
            self.reference.cycles.transparent_max_bounces = 24
            self.reference.cycles.max_bounces = 24
            self.reference.render.tile_x = 64
            self.reference.render.tile_y = 64
            self.reference.render.resolution_x = 400
            self.reference.render.resolution_y = 600
        # self.reference.render.use_persistent_reference = True


    def add_object_fixed(self, obj):
        ''' Add object in the list of fixed objects 
            Parameter:
                obj: BlenderObject
                    add the BlenderObject class object in the list of fixed objects
        '''
        self.objects_fixed.append(obj)

    def add_object_unfixed(self, obj):
        ''' Add object in the list of un-fixed objects
            Parameter:
                obj: BlenderObject
                    add the BlenderObject class object in the list of un-fixed objects
        '''
        self.objects_unfixed.append(obj)

    def import_object(self, filepath, location=(0,0,0), scale=(1,1,1), \
            orientation=(0,0,0), overwrite=True, fixed=False):
        ''' Importing object from the file along with their geometrical config 
            Parameters:
                filepath: string
                    Import 3d model from the file, currenlty supported model extensions: .obj, .dae
                location: tuple(float, float, float), optional
                    set the location of the imported model to the `location`
                scale: tuple(float, float, float), optional
                    set the scale of the imported 3d model to the `scale`
                orientation: tuple(float, float, float), optional
                    set the orientation of the imported 3d model to the `orientation`
                overwrite: bool, optional
                    If False, the model original configuration(written in model file) is kept unchanged. Default is True
                fixed: bool, optional
                    If True, the object is then added to the list of fixed objects in the scene, otherwise fixed in the list of unfixed objects in the scene. Default is False
        '''
        new_obj = BlenderObject(filepath=filepath, scale=scale, \
            location=location, orientation=orientation, overwrite=overwrite)
        if not fixed:
            self.add_object_unfixed(new_obj)
        else : self.add_object_fixed(new_obj)
        return new_obj

    def delete_all(self, ):
        ''' Clears up everything from the scene, absolutely everything'''
        for obj in self.objects_fixed:
            obj.delete()
        for obj in self.objects_unfixed:
            obj.delete()
        for subject in self.subject:
            subject.delete()
        self.subject = []
        for subject in self.subject_bot:
            subject.delete()
        self.subjects = []
        self.subjects_bot = []
        self.remove_lambs
        self.objects_fixed = []
        self.objects_unfixed = []
        # doing it all again... all the objects are deleted by their respective delete() call
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        # deleting Orphan reference blocks and mesh material
        for obj in self.reference.objects:
            if obj.type == 'MESH':
                obj.select = True
            else :
                obj.select = False
        bpy.ops.object.delete()
        for block in bpy.reference.materials:
            if block.users == 0:
                bpy.reference.meshes.remove(block)
        for block in bpy.reference.textures:
            if block.users == 0:
                bpy.reference.meshes.remove(block)
        for block in bpy.reference.images:
            if block.users == 0:
                bpy.reference.meshes.remove(block)

    def remove_object(self, obj):
        ''' remove object from the scene and delete it's model data 
            Parameters:
                obj: BlenderObject
                    remove the BlenderClass object `obj` from the Scene and delete it's model data
        '''
        self.objects_unfixed.remove(obj)
        obj.delete()
 
    def render_to_file(self, file_path):
        ''' save a rendered image to a file
            Parameters:
                filepath: string
                    relative or absolute file path to store the rendered image of the scene
        '''
        self.reference.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

    #TODO: NOT NEEDED
    def add_background(self, background):
        self.background = backgroud

    #TODO: Update
    def add_camera(self, camera):
        self.camera = camera

    #TODO: Not Needed
    def add_subject(self, subject, subject_bot=None):
        self.subjects.append(subject)
        self.subjects_bot.append(subject_bot)

    #TODO update
    def add_lambs(self, lamb):
        self.lamb.append(lamb)

    #TODO: Not Needed
    def remove_subject(self, ):
        for subject in self.subjects:
            subject.delete()
        for subject_bot in self.subjects_boy:
            subject_bot.delete()
        self.subjects = []
        self.subjects_bot = []
    #TODO: might not need
    def remove_lamps(self, ):
        for lamp in self.lamps:
            lamp.delete()
        self.lamps = []

    def point_conversion(self, x, y , z):
        ''' convert the 3d to  2d point from the perspective of camera of the scene
            Parameters:
                x, y, z: float, float, float
                    (x, y, z) represents the location of the point in the 3d space
            Returns:
                x, y, z: int, int, int
                    (x, y) is the point in 2d space in perspective of the scene camera (self.caemra)
                    -ve `z` means the point in 3d space is not in view of camera because of being behind the scene camera
                    +ve `z` is just a normal point in view of camera in 3d space
                    Note: Point could be out of view, the tuple (x, y) will be accordingly more positive than image dimensions
                          or negative in case the points are above of view or left of scene camera view
        '''
        if self.camera is None:
            raise Exception('No camera found in the scene for rendering')
        bpy.context.scene.cursor.location = (x, y, z)
        co_2d = bpy_extras.object_utils.world_to_camera_view(
            self.reference,
            self.camera.reference,
            bpy.context.scene.cursor.location)

        # If you want pixel coords
        render_scale = self.reference.render.resolution_percentage / 100
        render_size = (
            int(self.reference.render.resolution_x * render_scale),
            int(self.reference.render.resolution_y * render_scale),
        )
        x = round(co_2d.x * render_size[0], 6)
        y = round(co_2d.y * render_size[1], 6)
        z = round(co_2d.z * render_size[1])
        return x, y, z
        # z is just for the placement of the object from the camera
        # if z is positive then the object is in front, 
        # and if negative then the object is behind the camera and not visible

    def get_annotation(self, ):
        '''
        Generate annotations for the object in the list of unfixed objects in the scene
        Note: Don't know to which extent this works... But what it is doing is that it's projecting every point of the bouding box of the object in 2d space, and then finding the min/max of x and y would lead to the bouding box in 2d space.
              concern is that the 3d bounding box isn't efficient, and should be replaced to a more efficient bounding box which I believe is available in blenderPY #TODO need to update this
        return:
            dict():
                dict structure is as follows:
                annotation = {'object_name': [(top_left_x, top_left_y), (right_bottom_x, right_bottom_y)],
                              ... ,
                             }
        '''
        annotation = {}
        for obj in self.objects_unfixed:
            dim_x = obj.reference.dimensions.x
            dim_y = obj.reference.dimensions.y
            dim_z = obj.reference.dimensions.z
            x, y, z = obj.get_location()
            xs = [x-dim_x/2, x+dim_x/2]
            ys = [y-dim_y/2, y+dim_y/2]
            zs = [z-dim_z/2, z+dim_z/2]
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            for ix in xs:
                for iy in ys:
                    for iz in zs:
                        _x, _y, _z = self.point_conversion(ix, iy, iz)
                        if _x > max_x: max_x = _x
                        if _x < min_x: min_x = _x
                        if _y > max_y: max_y = _y
                        if _y < min_y: min_y = _y
            annotation[obj.name] = [(round(min_x), round(max_y)), (round(max_x),round(min_y))]
        return annotation
