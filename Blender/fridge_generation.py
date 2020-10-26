import bpy
import sys
import math

#add folder containing utils to sys paths for importing
sys.path.append(r"C:\Users\nickh\Documents\GitHub\Fridge-Food-Type\Blender")
from blender_automation.utils import select_object, clean_sketchup_cams, delete_all, find_z_coord, intersection_check

delete_all()

#Import fridge 3d model
fridge_path = r"C:\Users\nickh\Google Drive\Colab Notebooks\fridge_det\Blender\objects\fridge_base.dae"
bpy.ops.wm.collada_import(filepath=fridge_path)

#Add Wine bottle
wine2_loc = r"C:\Users\nickh\Google Drive\Colab Notebooks\fridge_det\Blender\objects\750ML_Wine\750ML_Wine.obj"
bpy.ops.import_scene.obj(filepath=wine2_loc)
select_object('750ML_Wine')
bpy.context.view_layer.objects.active = bpy.data.objects['750ML_Wine']
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.transform.resize(value=(0.014387, 0.014387, 0.014387), orient_type='GLOBAL')
bpy.context.object.rotation_euler = (0, 0, 0)
z = find_z_coord('750ML_Wine', origin_center=True, shelf_num=1)
bpy.context.object.location = (0, -.05, z)

#Add apple
apple_loc = r"C:\Users\nickh\Google Drive\Colab Notebooks\fridge_det\Blender\objects\apple\manzana2.obj"
bpy.ops.import_scene.obj(filepath=apple_loc)
apple_obj = bpy.context.selected_objects[0]
bpy.data.objects['manzana2'].select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects['manzana2']
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.transform.resize(value=(0.001365, 0.001365, 0.001365), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
bpy.context.object.rotation_euler = (-0.778549, -0.057743, 0.137881)
apple_z = find_z_coord('manzana2')
bpy.context.object.location = (0.19139, -0.10539, apple_z)


#Tomato code chunk
tomato_loc = r"C:\Users\nickh\Google Drive\Colab Notebooks\fridge_det\Blender\objects\tomato\Tomato_v1.obj"
bpy.ops.import_scene.obj(filepath=tomato_loc)
select_object('Tomato_v1')
bpy.context.view_layer.objects.active = bpy.data.objects['Tomato_v1']
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.transform.resize(value=(0.009365, 0.009365, 0.009365), orient_type='GLOBAL')
bpy.context.object.rotation_euler = (-0.175, 0, 0)
tomato_z = find_z_coord('Tomato_v1')
bpy.context.object.location = (0, -.3, tomato_z)

#Grapes
loc = r"C:\Users\nickh\Google Drive\Colab Notebooks\fridge_det\Blender\objects\grapes_1\grapes_1.obj"
bpy.ops.import_scene.obj(filepath=loc)
select_object('grapes_1')
bpy.context.view_layer.objects.active = bpy.data.objects['grapes_1']
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.ops.transform.resize(value=(0.02, 0.02, 0.02), orient_type='GLOBAL')
bpy.context.object.rotation_euler = (0, math.radians(45), math.radians(45))
z = find_z_coord('grapes_1', origin_center=True, shelf_num=2)
bpy.context.object.location = (-0.1, -.3, z)

#lettuce
let_loc = r"C:\Users\nickh\Google Drive\Colab Notebooks\fridge_det\Blender\objects\LettuceRomaine\LettuceRomaine.obj"
bpy.ops.import_scene.obj(filepath=let_loc)
select_object('LettuceRomaine')
bpy.context.view_layer.objects.active = bpy.data.objects['LettuceRomaine']
bpy.ops.transform.resize(value=(0.008, 0.008, 0.008))
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
bpy.context.object.rotation_euler = (math.radians(180), 0, 0)
let_z = find_z_coord('LettuceRomaine', origin_center=True, shelf_num=1)
bpy.context.object.location = (-0.12, -0.3, let_z)


#Adding camera
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, -2.25, 2), rotation=(1.26929, 0 , 0), scale=(1, 1, 1))
#Adding light
bpy.ops.object.light_add(type='POINT', radius=0.25, align='WORLD', location=(0, -1, 1.5), scale=(1, 1, 1))


# Setting render settings - only some render engines are compatible with used textures/materials
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.camera = bpy.data.objects['Camera']
bpy.context.scene.render.resolution_x = 400
bpy.context.scene.render.resolution_y = 800
bpy.context.scene.render.filepath = '/Users/nickh/Documents/Fellowship/Fridge/3D generation/output.jpg'
#bpy.ops.render.render(write_still=True)

print(intersection_check('manzana2'))


''' 
can be executed as :
blender --background --python fridge.py #remove the background to actually see the blender GUI following the python script
'''
