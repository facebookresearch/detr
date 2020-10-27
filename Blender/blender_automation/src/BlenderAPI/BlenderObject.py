import bpy
import mathutils as mathU
import os
import pandas as pd
import random

#Json generated from jupyter NB, imports as dataframe
#index by object name
#available params are shelves, path, origin, scale_factor
#Not sure best place to put this line
object_dict = pd.read_json('object_dict.json')

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

def intersection_check(checked_obj):
    """
    Will take item name and output if it intersects with any other objects.
    Returns True if intersecting an object, false if no intersections. 
    """
    scene =  bpy.context.scene
    for obj_next in bpy.context.scene.objects:
        if obj_next.type == 'MESH':
            obj_name = obj_next.name
            #initialize bmesh objects
            bm1 = bmesh.new()
            bm2 = bmesh.new()
            #fill bmesh data from objects
            bm1.from_mesh(scene.objects[checked_obj].data)
            bm2.from_mesh(scene.objects[obj_name].data)            
            #transform needed to check inter
            bm1.transform(scene.objects[checked_obj].matrix_world)
            bm2.transform(scene.objects[obj_name].matrix_world) 
            #make BVH tree from BMesh of objects
            obj_BVHtree = BVHTree.FromBMesh(bm1)
            obj_next_BVHtree = BVHTree.FromBMesh(bm2)           

            #get intersecting pairs
            inter = obj_BVHtree.overlap(obj_next_BVHtree)

    if inter != []:
        return True
    else:
        return False

def find_z_coord(item_name, origin_center=True, shelf_num=1):
    """
    Gives the necessary z-axis coordinate to place item on shelf.
    """
    shelf_heights = [1.2246, 1.5581, 1.7443]
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
        else :
            self.reference = reference
        self.name = self.reference.name

    def blender_create_operations(self, ):
        raise Exception('Error: Not yet implemented')

    def set_location(self, x, y, z):
        ''' set location for the object in the scene '''
        self.reference.location=(x, y, z)
        print(f'{self.name} : location set to : {(x, y, z)}')

    def place_randomly(self, ):
        """
        Chooses random coords for object placement. 
        Requires dict of object states. 
        Should be run after resize is done.
        """
        item_params = object_dict[self.name]
        dim_x, dim_y, dim_z = bpy.data.objects[self.name].dimensions
        x_lims = [-0.25493 + dim_x/2, 0.29941 - dim_x/2]#hardcoded limits, offset by item width
        y_lims = [-0.4206 + dim_y/2, 0.025957 - dim_y/2]#hardcoded limits, offset by item width
        retry_tracker = True
        while retry_tracker == True:
            z_temp = find_z_coord(self.name, origin_center=item_params['origin']=='CENTER', shelf_num=random.choice(item_params['shelves']))
            x_temp = random.uniform(x_lims[0], x_lims[1])
            y_temp = random.uniform(y_lims[0], y_lims[1])
            self.set_location(x=x_temp, y=y_temp, z=z_temp)
            retry_tracker = intersection_check(self.name)

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
        q = q*self.reference.rotation_quaternion
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

