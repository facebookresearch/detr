import bpy
import mathutils as mathU
import os

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
        q = to_quternion(w, x, y, z)
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

