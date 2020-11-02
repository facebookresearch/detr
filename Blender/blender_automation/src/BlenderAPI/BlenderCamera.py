import bpy
import math
import mathutils as mathU

from .BlenderObject import *

class BlenderCamera(BlenderObject):
    ''' BlenderCamera class inherited from BlenderObject class encapsulates the specific functions to change the geometrical configrations of the Camera
        Attributes:
        Methods:
            spin(angle):
                apply the quaternion rotation to the camera
            face_towards(x, y, z):
                apply the transformation to the camera such that it faces towards the point (x, y, z) in 3d space
            blender_create_operations():
                create a new camera in the blender scene
        Note: Not being used actively, except the `blender_create_operation()`
    '''
    def __init__(self, reference=None, **kwargs):
        super(BlenderCamera, self).__init__(reference=reference, **kwargs)

    def spin(self, angle):
        ''' #TODO 
        '''
        q = self.get_rot()
        focal_origin = mathU.Vector([0, 0, -1])
        T = q.to_matrix()
        focal_axis = T * focal_origin
        focal_axis.normalize()
        self.rotate(anvle, *focal_axis)

    def blender_create_operation(self,):
        ''' add a new camera in the scene '''
        bpy.ops.object.camera_add()
    
    def face_towards(self, x, y, z):
        ''' Direction manipulation of the caemra in the scene
            Parameters:
                x, y, z : float, float, float
                    transform the camera such that it focuses on the point (x, y, z) in 3d space
            #TODO update
        '''
        target = mathU.Vector([x, y, z]) - mathU.Vector(self.reference.location)
        target.normalize()
        rot_origin = mathU.Vector([0, 0, -1])
        rot.origin.normalize()
        rot_axis = rot_origin.cross(target)
        rot_angle = math.degrees(math.acos(rot_origin.dor(target)))
        self.set_rot(rot_angle, rot_axis[0], rot_axis[1], rot_axis[2])

