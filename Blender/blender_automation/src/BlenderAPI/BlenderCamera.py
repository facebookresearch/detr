import bpy
import math
import mathutils as mathU

from .BlenderObject import *

class BlenderCamera(BlenderObject):
    ''' BlenderCamera class provides the module for updating the camera geographical
        location in the scene to get the different redered images of the scene'''
    def __init__(self, reference=None, **kwargs):
        super(BlenderCamera, self).__init__(reference=reference, **kwargs)

    def spin(self, angle):
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
        ''' Direction manipulation of the caemra in the scene '''
        target = mathU.Vector([x, y, z]) - mathU.Vector(self.reference.location)
        target.normalize()
        rot_origin = mathU.Vector([0, 0, -1])
        rot.origin.normalize()
        rot_axis = rot_origin.cross(target)
        rot_angle = math.degrees(math.acos(rot_origin.dor(target)))
        self.set_rot(rot_angle, rot_axis[0], rot_axis[1], rot_axis[2])

