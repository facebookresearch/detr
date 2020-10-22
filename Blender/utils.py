#Blender utilities functions to simplify scripting of images
import bpy

def select_object(obj_name, has_children=True):
    """
    selects the object specified by object name. 
    If has_children is True (defaults to true), will select children and 1 layer of subchildren
    """
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[obj_name].select_set(True)
    if has_children:
        for child in bpy.data.objects[obj_name].children:
            child.select_set(True)
            for subchild in child.children:
                subchild.select_set(True)
                
#Sketchup imports include last camera view. We do not want these.
def clean_sketchup_cams():
    """
    Cleans sketchup cams from imported objects. 
    Sketchup files include last camera view and sometimes a camera preview.
    """
    bpy.ops.object.select_all(action='DESELECT')
    if bpy.context.scene.objects.get('skp_camera_Last_Saved_SketchUp_View'):
        bpy.data.objects['skp_camera_Last_Saved_SketchUp_View'].select_set(True)
    if bpy.context.scene.objects.get('skp_camera_Preview'):
        bpy.data.objects['skp_camera_Preview'].select_set(True)
    bpy.ops.object.delete()
    
def delete_all():
    """
    resets the context, selects all objects and deletes them. 
    Just selecting all objects and deleting will leave orphan datablocks.
    """
    bpy.context.scene #reset context to be able to clear objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True) # syntax for currently latest version of blender
            #obj.select = True # deprecated syntax, was used in earlier versions of Blender
    bpy.ops.object.delete()

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def find_z_coord(item_name, origin_center=True, shelf_num=1):
    """
    Gives the necessary z-axis coordinate to place item on shelf.
    """
    if shelf_num == 1:
        shelf_z = 1.226
    elif shelf_num == 2:
        shelf_z = 1.56
    elif shelf_num == 3:
        shelf_z = 1.75 
    else:
        raise ValueError('shelf_num must be 1, 2, or 3')
    z_coord = shelf_z
    if origin_center:
        z_coord += bpy.data.objects[item_name].dimensions.z/2
    return z_coord


