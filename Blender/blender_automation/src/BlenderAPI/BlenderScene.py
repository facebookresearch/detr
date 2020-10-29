import bpy
import bpy_extras

# relative import from other classes
from .BlenderObject import *

class BlenderPlane(BlenderObject):
    def __init__(self, **kwargs):
        super(BlenderPlane, self).__init__(**kwargs)

    def blender_create_operation(self, ):
        bpy.ops.mesh.primitive_plane_add()

class BlenderRoom(object):
    def __init__(self, radius):
        self.walls = []
        self.walls.append(BlenderPlane(location=(-radius, 0, 0), scale=(radius, radius, radius), orientation=(90, 0, 1, 0)))
        self.walls.append(BlenderPlane(location=(0, radius, 0), scale=(radius, radius, radius), orientation=(90, 1, 0, 0)))
        self.walls.append(BlenderPlane(location=(0, 0, -radius), scale=(radius, radius, radius)))
        self.walls.append(BlenderPlane(location=(radius, 0, 0), scale=(radius, radius, radius), orientation=(90, 0, 1, 0)))
        self.walls.append(BlenderPlane(location=(0, -radius, 0), scale=(radius, radius, radius), orientation=(90, 1, 0, 0)))
        self.walls.append(BlenderPlane(location=(0, 0, radius), scale=(radius, radius, radius)))

    def delete(self, ):
        for wall in self.walls:
            wall.delete()
        self.walls = []

class BlenderScene(object):
    ''' BlenderScene object handles the scene in blender, takes care of importing objects, managing them as a part of the scene
        and manages the rendering configuration of the scene in blender from the class method set_render() '''
    def __init__(self, reference):
        self.lambs = []
        self.background = None # not in use
        self.objects_fixed = [] 
        self.objects_unfixed = []
        self.camera = None # not in any use in particular

        # self.subjects = []
        # self.subjects_bot = []
        # [NEED TO BE RESOLVED] : Note that we might not need the subjects and objects seperated, as we are considering all the items in the scene as objects.
        # Thinking about having the fridge as Fixed object and rest are the un-fixed objects...
        
        self.reference = reference #bpy.data.scenes[0] or bpy.context.scene

    def set_render(self, resolution=300, samples=128, set_high_quality=False):
        ''' Setting up the scene rendering configuration '''
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
        ''' Add object in the list of fixed objects '''
        self.objects_fixed.append(obj)

    def add_object_unfixed(self, obj):
        ''' Add object in the list of un-fixed objects '''
        self.objects_unfixed.append(obj)

    def import_object(self, filepath, location=(0,0,0), scale=(1,1,1), \
            orientation=(0,0,0), overwrite=True, fixed=False):
        ''' Importing object from the file along with their geometrical config '''
        new_obj = BlenderObject(filepath=filepath, scale=scale, \
            location=location, orientation=orientation, overwrite=overwrite)
        if not fixed:
            self.add_object_unfixed(new_obj)
        else : self.add_object_fixed(new_obj)
        return new_obj

    def delete_all(self, ):
        ''' Clears up everything from the scene'''
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
        ''' remove object from the scene and delete it's model data '''
        self.objects_unfixed.remove(obj)
        obj.delete()
 
    def render_to_file(self, file_path):
        ''' save a rendered image to a file '''
        self.reference.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

    # TODO: all the modules below haven't been in use yet... and might need update before using
    # TODO: NOT NEEDED
    def add_background(self, background):
        self.background = backgroud

    # TODO: Update
    def add_camera(self, camera):
        self.camera = camera

    # TODO: Not Needed
    def add_subject(self, subject, subject_bot=None):
        self.subjects.append(subject)
        self.subjects_bot.append(subject_bot)


    def add_lambs(self, lamb):
        self.lamb.append(lamb)

        # TODO: Not Needed
    def remove_subject(self, ):
        for subject in self.subjects:
            subject.delete()
        for subject_bot in self.subjects_boy:
            subject_bot.delete()
        self.subjects = []
        self.subjects_bot = []
    
    def remove_lamps(self, ):
        for lamp in self.lamps:
            lamp.delete()
        self.lamps = []

    def point_conversion(self, x, y , z):
        ''' convert the 3d to  2d point from the perspective of camera of the scene '''
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
