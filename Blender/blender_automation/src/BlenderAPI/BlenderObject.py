import bpy, bmesh
import mathutils as mathU
from mathutils.bvhtree import BVHTree
from math import pi
import os
import random
from time import time

def check_is_iter(input, size):
    ''' Function to check whether the first argument is a iterable or not, with the size of second argument
        parameters :
            input: object
                object which need to be checked, could be anything
            size : int
                size against which the object iterability is going to be checked
        return :
            True if `input` is iterable and contains `size` number of elements/objects
    '''
    try:
        input_iter = iter(input)
        return len(input) == size
    except TypeError:
        return False

def check_vector_non_negative(input):
    ''' Function to check the input is negative or not.
        parameters :
            input : vector or any iterable object
                vector whose values are in question
        return :
            True if `input` is iterable/vector like object consists of positive values only
    '''
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

def is1DOverlap(min1, max1, min2, max2):
    if min1 >= max2 or min2 >= max1: return False
    else: return True

# TL contains is minimum of (x,y) and RB is maximum(x, y) in ractangle

def is2DOverlap(min1, max1, min2, max2):
    if len(min1) != 2 or len(max1) != 2 or len(min2) != 2 or len(max2) != 2:
        raise Exception('Argument Expected 2d coordinates of left top points(min_x, min_y) and right bottom \
                         points(max_x, max_y) of two rectangle, make sure the arguments are like min1, max1, min2, max2. \
                         Got{min1, max1, min2, max2}')
    if is1DOverlap(min1[0], max1[0], min2[0], max2[0]) and \
       is1DOverlap(min1[1], max1[1], min2[1], max2[1]): return True
    else return False

def is3DOverlap(min1, max1, min2, max2):
    if len(max1) != 3 or len(max1) != 3 or len(min2) != 3 or len(max2) != 3:
        raise Exception('Argument Expected 3d coordinates of left top points(min_x, min_y, min_z) and right bottom \
                         points(max_x, max_y, max_z) of two rectangle, make sure the arguments are like min1, max1, min2, max2. \
                         Got{min1, max1, min2, max2}')
    if is1Doverlap(min1[0], max1[0], min2[0], max2[0]) and \
       is1Doverlap(min1[1], max1[1], min2[1], max2[1]) and \
       is1Doverlap(min1[2], max1[2], min2[2], max2[2]) : return True
    eles: return False

# Default shelf heights match only current fridge item. 
def find_z_coord(item_name, origin_center=True, shelf_num=1, shelf_heights=[1.2246, 1.5581, 1.7443]):
    """
    Gives the necessary z-axis coordinate to place item on shelf.
        parameters :
            item_name: string
                item/object name in the scene
            origin_center: bool
                [To be completed]
            shelf_num : int
                shelf number determines the shelf that is being used and according to that which height is to be used
            shelf_heights: array/list[float, ]
                array containing the shelf hieghts of different shelves in the fridge(in our case) otherwise depends on the model being used
        returns : float
            height/z coordinate accoriding to the height of the shelf being used for placement of the objects
    """
    shelf_z = shelf_heights[shelf_num-1]
    z_coord = shelf_z
    if origin_center:
        z_coord += bpy.data.objects[item_name].dimensions.z/2
    return z_coord

class BlenderObject(object):
    '''  Base class representing a Blender Object includes imported model, scene, camera, etc. It also encapsulates different methods that are frequently used for blender objects, including simple operations like setting location, scale, orientation
         Attributes:
             filepath : string (optional)
                 Given a BlenderObject refering to an imported model, `filepath` would be storing the file path of the model imported.
             reference : bpy objects (optional)
                 reference that blender uses for the object. Ex: bpy.data.objects['model_X'] or bpy.context.scene.objects[0]
             name : string (optional)
                 name of the imported model which could be used for refering the actual bpy objects like `bpy.data.objects['name']
             parent : string (optional)
                 name of the imported model in case an object is imported multiple times, then the original name would be require to fetch the object parameters like rotation lock, which shelf to be used, NOTE: highly specific to our use case that is placement in the fridge

        Methods:
            __init__(location=(), orientation=(), scale=(), refernece=None, name=None, filepath=None, overwrite=True, **kwargs):
                Construct all the necessary objects and invokes suitable modules to create a object of BlenderObject class
            blender_create_operation():
                Implemented in child classes where the object needs to be created from bpy directly like light and camera
            set_location(x, y, z):
                set the actual location of the object in the scene
            set_scale((x, y, z)):
                set the actual scale to the object in the scene
            set_euler_rotation(x, y, z):
                set euler roation to the object in the scene
            set_rotation(w, x, y, z):
                set rotation in quaternion mode
            get_rotation():
                return quaternion roation of the object
            get_location():
                return absolute location of the object in the scene
            get_scale():
                return the absolute scale of the object in the scene
            get_euler_rotation():
                return the absolute euler orientation parameters of the object
            delete():
                delete all the object associate data from the blender runtime including orphan data blocks left after blender deleting object
            place_randomly(params):
                Find a place in the scene for the object randomly based on uniform distribution
            load_from_file():
                Load object from the file, currenlty supported format: .obj, .dae
            is_intersection():
                check if the this object is intersecting from other object in the scene which are of type MESH and not of the name `refrigerator`
            rotate(w, x, y, z):
                rotate the object from it's current orientation to the one corresponding the transformation from (`x`, `y`, `z`)
    '''
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
        self.parent = self.name.split('.')[0]

    def blender_create_operations(self, ):
        ''' Implemented in child classes which actually use this feature. Example: light, camera '''
        raise Exception('Error: Not yet implemented')

    def set_location(self, x, y, z):
        ''' set location for the object in the scene `set_location(x, y, z)`
            parameters:
                x, y, z: float, float, float
                    set the actual location of the object to (x, y, z) in the 3d space
        '''
        self.reference.location=(x, y, z)
        print(f'{self.name} : location set to : {(x, y, z)}')

    def get_location(self, ):
        ''' returns the absolute location of the objects
            returns:
                (x, y, z): (float, float, float)
                    3d coordinate of the object location
        '''
        return self.reference.location

    def set_euler_rotation(self, x, y, z):
        ''' set euler orientation for the object in the scene `set_euler_rotation(x, y, z)`
            parameters:
                x, y, z: float, float, float
                    set the actual euler orientation of the object to x, y, z
        '''
        self.reference.rotation_mode = 'XYZ'
        self.reference.rotation_euler = (x, y ,z)
        print(f'{self.name} : euler rotation set to {(x, y, z)}')

    def set_rotation(self, w, x, y, z):
        ''' set quaternion rotation for the object in the scene `set_rotation(w, x, y, z)
            parameters:
                w, x, y, z: float, float, float, float
                    set the actual quaternion orientation of the object to w, x, y, z
        '''
        self.reference.rotation_mode = 'QUATERNION'
        q = to_quaternion(w, x, y, z)
        self.reference.rotation_quaternion = q

    def set_scale(self, scale):
        ''' set scale for the object in the scene `set_scale(_scale_)
            parameters:
                scale: iterable of size 3
                    set the scale of the object to the (scale[0], scale[1], scale[2])
        '''
        valid = check_is_iter(scale, 3) and check_vector_non_negative(scale)
        if not valid:
            raise Exception(f'Scale input is Invalid')
        self.reference.scale = scale
        print(f'{self.name} : Scale set to {scale}')

    def get_euler_roation(self, ):
        ''' get the euler orientation parameters
            returns:
                absolute euler rotation of the object / bpy.data.objects['xxx'].rotation_euler
        '''
        return self.reference.rotation_euler
    
    def apply_rotation(self, ):
        '''Force rotation changes to be transferred from object rotation to global position change. 
            Needed for updated bounding boxes.
        '''
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[self.name].select_set(True)
        for child in bpy.data.objects[self.name].children:
            child.select_set(True)
            for subchild in child.children:
                subchild.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    def get_rotation(self, ):
        ''' get quaternion orientation parameters
                returns:
                    quaternion orientation of the object / bpy.data.object['xxx'].rotation_quaternion
        '''
        return self.reference.rotation_quaternion

    def get_scale(self, ):
        ''' get scale parameters for the object
            returns:
               scale of the object
        '''
        return self.reference.scale

    def rotate(self, w, x, y, z):
        ''' Rotate the object in quaternion format 
            parameter:
                w, x, y, z: float, float, float, float
                    rotate the object in the quaternion format corresponding the transformation from w, x, y, z
        '''
        self.reference.roation_mode = 'QUATERNION'
        q = to_quaternion(w, x, y, z)
        q = q * self.reference.rotation_quaternion
        self.reference.rotation_quaternion = q

    def delete(self):
        ''' Delete the object including orphan data blocks associated with the object'''
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
        ''' Import Object from the file, using filepath given while intializing the class object'''
        if not os.path.isfile(os.path.abspath(self.filepath)):
            raise Exception(f'Object File does not exists: {self.filepath}')
        if self.filepath[-4:] == '.dae':
            bpy.ops.wm.collada_import(filepath=self.filepath)
        elif self.filepath[-4:] == '.obj':
            bpy.ops.import_scene.obj(filepath=self.filepath)

    def is_intersecting(self, type='BOUNDING_BOX'):
        ''' Check whether the object is intersecting to any other object in the scene or not
            return: bool
               True : if object is intersecting/overlapping with other object of type 'MESH' in the scene
               False : otherwise
        '''
        if not self.reference.type == 'MESH':
            raise Exception('Object is of not MESH type, hence can\'t find intersection with other MEST type objects')

        #mesh matrix doesn't automatically update. Force update at beginning of call
        bpy.context.view_layer.update()

        start = time()

        if type == 'BOUNDING_BOX':
            self_bound = self.reference.bound_box
            _xyz = (self_bound[0][0], self_bound[0][1], self_bound[0][2])
            # positive_negative : example x_yz : x is positive and y,z are negative
            # z_xy = (self_bound[1][0], self_bound[1][1], self_bound[1][2])
            # yz_x = (self_bound[2][0], self_bound[2][1], self_bound[2][2])
            # y_xz = (self_bound[3][0], self_bound[3][1], self_bound[3][2])
            # _xyz = (self_bound[4][0], self_bound[4][1], self_bound[4][2])
            # xz_y = (self_bound[5][0], self_bound[5][1], self_bound[5][2])
            # xyz_ = (self_bound[6][0], self_bound[6][1], self_bound[6][2])
            # xy_z = (self_bound[7][0], self_bound[7][1], self_bound[6][2])
            self_min_xyz = (self_bound[0][0], self_bound[0][1], self_bound[0][2])
            self_max_xyz = (self_bound[6][0], self_bound[6][1], self_bound[6][2])
            for obj in bpy.context.scene.objects:
                if obj.type != 'MESH' or obj.name == self.name or 'refrigerator' in obj.name: continue
                obj_bound = obj.reference.bound_box
                obj_min_xyz = (obj_bound[0][0], obj_bound[0][1], obj_bound[0][2])
                obj_max_xyz = (obj_bound[6][0], obj_bound[6][1], obj_bound[6][2])
                if is3DOverlap(self_min_xyz, self_max_xyz, obj_min_xyz, obj_max_xyz):
                    end = time()
                    print(f'[{self.name}] Intersection Found with {obj.name}  Time Elapsed: {(end-start)} seconds')
                    return True
        else :
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

    def place_randomly(self, params):
        """
        Chooses random coords for object placement. 
        Requires dict of object states. 
        Should be run after resize is done.
        parameters:
            params: dict
                attribute required for an object to be placed randomly, [To be completed]
        """
        #rotate around unconstrained axis. Will need more robust logic for more than 1 axis
        #Have to force scene update to get new x/y dimensions post rotation, want to avoid in loop for performance
        rotation = [axis*random.uniform(0, 2*pi) for axis in params['unconstrained_axis']]
        self.set_euler_rotation(rotation[0], rotation[1], rotation[2])
        self.apply_rotation()
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
