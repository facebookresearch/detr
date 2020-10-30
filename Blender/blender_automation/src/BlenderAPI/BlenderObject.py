import bpy, bmesh
import mathutils as mathU
from mathutils.bvhtree import BVHTree
from math import pi
import os
import random
from time import time

def check_is_iter(input, size):
    try:
        input_iter = iter(input)
        return len(input) == size
    except TypeError:
        return False

def check_vector_non_negative(input):
    for item in input:
        if not (item >= 0):
            return False
    return True

def rotate(vector, quaternion):
    """
    utility function to rotate a vector, given a rotation in the form of a quaternion
    :param vector: vector to rotate
    :param quaternion: rotation in the form of quaternion
    :return : rotated vector
    """
    vecternion = mathU.Quaternion([0, vector[0], vector[1], vector[2]])
    quanjugate = quaternion.copy()
    quanjugate.conjugate()
    return quaternion * vecternion * quanjugate

# Default shelf heights match only current fridge item. 
def find_z_coord(item_name, origin_center=True, shelf_num=1, shelf_heights=[1.2246, 1.5581, 1.7443]):
    """
    Gives the necessary z-axis coordinate to place item on shelf.
    """
    shelf_z = shelf_heights[shelf_num-1]
    z_coord = shelf_z
    if origin_center:
        z_coord += bpy.data.objects[item_name].dimensions.z/2
    return z_coord

class BlenderObject(object):
    '''  Object class to handle all blender objects, provides method to change their geometrical configuration, and importing
        or deleting them'''
    def __init__(self, location=(0,0,0), orientation=(0,0,0), scale=(1,1,1), reference=None, name=None, filepath=None, overwrite=True, **kwargs):
        ''' constructor to initiate the BlenderObject object with reference(bpy.data.object[0]) or just by name, or direcly
            import the object from the file passed as an argument'''
        if reference is None:
            if filepath is not None:
                ''' Import the object from the file '''
                self.filepath = filepath
                self.load_from_file()
                self.reference = bpy.context.selected_objects[0]
                self.name = self.reference.name
                bpy.context.view_layer.objects.active = self.reference
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                if overwrite: # some objects are better imported in their default state, but some aren't so here it is
                    self.set_location(*location)
                    self.set_euler_rotation(*orientation)
                    self.set_scale(scale)

            else:
                if name is not None:
                    'Selects the object specified by the object name. If has_childern is True(default is True), will select children and 1 layer of subchildren'
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.data.objects[name].select_set(True)
                    for child in bpy.data.objects[name].children:
                        child.select_set(True)
                        for subchild in child.children:
                            subchild.select_set(True)
                # will take reference for whichever object is selected in the context
                print(bpy.context.selected_objects)
                assert len(bpy.context.selected_objects) == 1, "more than one object are selected!!!"
                self.reference = bpy.context.selected_objects[0]
        else:
            self.reference = reference
        self.name = self.reference.name

    def blender_create_operations(self, ):
        raise Exception('Error: Not yet implemented')

    def set_location(self, x, y, z):
        ''' set location for the object in the scene '''
        self.reference.location=(x, y, z)
        print(f'{self.name} : location set to : {(x, y, z)}')

    def get_location(self, ):
        return self.reference.location

    def place_randomly(self, params):
        """
        Chooses random coords for object placement. 
        Requires dict of object states. 
        Should be run after resize is done.
        """
        #rotate around unconstrained axis. Will need more robust logic for more than 1 axis
        #Have to force scene update to get new x/y dimensions post rotation, want to avoid in loop for performance
        #Moved rotation such that it's tried once per object then only coordinates are changed
        rotation = [axis*random.uniform(0, 2*pi) for axis in params['unconstrained_axis']]
        self.set_euler_rotation(rotation[0], rotation[1], rotation[2])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        #Bounding box vertices stored in 8*3 array. Object based coordinates, need to apply scale
        #1st 3 number array is low_x, low_y, low_z
        #7th 3 number array is high_x, high_y, high_z
        bound_box = bpy.data.objects[self.name].bound_box
        dim_x = (bound_box[6][0] - bound_box[0][0]) * params['scale_factor']
        dim_y = (bound_box[6][1] - bound_box[0][0]) * params['scale_factor']
        #hardcoded numbers for shelf width of base fridge model, offset by item width
        x_lims = [-0.25493 + dim_x/2, 0.29941 - dim_x/2]
        #hardcoded numbers for shelf width of base fridge model, offset by item width
        y_lims = [-0.4206 + dim_y/2, 0.025957 - dim_y/2]
        retry_tracker = True
        attempts = 0
        while retry_tracker == True and attempts < 9:
            print(f'\n{self.name} Dimensions:\n{self.reference.dimensions}')
            #Generate random X,Y,Z coords
            x_temp = random.uniform(x_lims[0], x_lims[1])
            y_temp = random.uniform(y_lims[0], y_lims[1])
            z_temp = find_z_coord(self.name, origin_center=params['origin']=='CENTER', shelf_num=random.choice(params['shelves']))
            self.set_location(x=x_temp, y=y_temp, z=z_temp)
            # calling a class method to check if the object is intersecting with others
            retry_tracker = self.is_intersecting()
            attempts +=1
            if attempts == 9:#place at origin if too many tries, removes from render for this image. 
                self.set_location(0,0,0)

    def set_euler_rotation(self, x, y, z):
        ''' set euler orientation for the object in the scene '''
        self.reference.rotation_mode = 'XYZ'
        self.reference.rotation_euler = (x, y ,z)
        print(f'{self.name} : euler rotation set to {(x, y, z)}')

    def set_rotation(self, w, x, y, z):
        ''' set quaternion rotation for the object in the scene '''
        self.reference.rotation_mode = 'QUATERNION'
        q = to_quaternion(w, x, y, z)
        self.reference.rotation_quaternion = q

    def set_scale(self, scale):
        ''' set scale for the object in the scene '''
        valid = check_is_iter(scale, 3) and check_vector_non_negative(scale)
        if not valid:
            raise Exception(f'Scale input is Invalid')
        self.reference.scale = scale
        print(f'{self.name} : Scale set to {scale}')

    def get_rot(self, ):
        ''' get quaternion orientation parameters '''
        return self.reference.rotation_quaternion

    def get_scale(self, ):
        ''' get scale parameters for the object '''
        return self.reference.scale

    def rotate(self, w, x, y, z):
        ''' Rotate the object in quaternion format '''
        self.reference.roation_mode = 'QUATERNION'
        q = to_quaternion(w, x, y, z)
        q = q * self.reference.rotation_quaternion
        self.reference.rotation_quaternion = q

    def delete(self):
        ''' Delete the object including orphan data associated with the object'''
        if self.reference is None:
            return
        bpy.ops.object.select_all(action='DESELECT')
        # (deprecated) self.reference.select = True
        self.reference.select_set(True)
        bpy.ops.object.delete()
        # Delete Orphan data
        try:
            if bpy.data.meshes[self.name].users == 0:
                bpy.data.meshes.remove(bpy.data.meshes[self.name])
            if bpy.data.material[self.name].users == 0:
                bpy.data.material.remove(bpy.data.material[self.name])
            if bpy.data.textures[self.name].users == 0:
                bpy.data.textures.remove(bpy.data.textures[self.name])
            if bpy.data.images[self.name].users == 0:
                bpy.data.images.remove(bpy.data.images[self.name])
        except:
            pass
        self.reference = None


    def load_from_file(self, ):
        ''' Import Object from the file'''
        if not os.path.isfile(os.path.abspath(self.filepath)):
            raise Exception(f'Object File does not exists: {self.filepath}')
        if self.filepath[-4:] == '.dae':
            bpy.ops.wm.collada_import(filepath=self.filepath)
        elif self.filepath[-4:] == '.obj':
            bpy.ops.import_scene.obj(filepath=self.filepath)

    def is_intersecting(self, ):
        #mesh matrix doesn't automatically update. Force update at beginning of call
        bpy.context.view_layer.update()
        if not self.reference.type == 'MESH':
            raise Exception('Object is of not MESH type, hence can\'t find \
                    intersection with other MEST type objects')
        start = time()
        #object calcs moved outside loop where possible
        bm1 = bmesh.new()
        bm1.from_mesh(self.reference.data)
        bm1.transform(self.reference.matrix_world)
        self_BVHtree = BVHTree.FromBMesh(bm1)
        for obj in bpy.context.scene.objects:
            if obj.name == self.name or not obj.type == 'MESH' or 'refrigerator' in  obj.name:
                continue
            # initialize bmesh objects
            bm2 = bmesh.new()
            # fill bmesh data from objects
            bm2.from_mesh(obj.data)
            # transform needed to check intersection
            bm2.transform(obj.matrix_world)
            # make BVH tree from BMesh of objects
            obj_BVHtree = BVHTree.FromBMesh(bm2)
            # get intersection
            intersection = self_BVHtree.overlap(obj_BVHtree)
            if intersection != []:
                end = time()
                print(f'[{self.name}] Intersection Found with {obj.name}  Time Elapsed: {(end-start)} seconds')
                return True
        end = time()
        print(f'[{self.name}] No Intersection Found  Time Elapsed: {(end-start)} seconds')
        return False