"""
Used to generate object dictionary for random placement.
shelves: shelves that object is allowed to be placed on (enum in 1,2,3)
path: path to object file within blender_automation/workspace directory
origin: possibly depriciated, where origin of item is stored. Should be CENTER for all .obj files?
scale_factor: scale factor to apply to object on import.
'import_rotations': if any rotations need to be performed on this object every time it's imported, put here
    -Euler rotation in (x,y,z) format
unconstrained axis: axis allowed for euler random rotation. Only 1 axis per item right now.
    Stored as T/F binary list (x, y, z)
label_cat: name of labeled category for annotations
    -enum in['n_apple', 'n_bell_pepper', 'n_bottle_juice',
            'n_bottle_milk', 'n_box_milk', 'n_can_soda', 'n_desserts',
            'n_eggs', 'n_grapes', 'n_hot_sauce', 'n_jam_jelly',
            'n_jar_food', 'n_ketchup', 'n_leafy_vegetable', 'n_lemon',
            'n_mustard', 'n_salad_dressing', 'n_tomato', 'n_water',
            'n_wine']
max_repeats: To be used - maximum times the object can be placed in an image. 
"""
import os
import json
from math import pi

os.chdir(r"C:\Users\nickh\Documents\GitHub\Fridge-Food-Type\Blender\blender_automation\src")
base_path = "C:/Users/nickh/Documents/GitHub/Fridge-Food-Type/Blender/blender_automation/workspace/objects/"
#Individual object parameters
wine = {
    'shelves': [1],
    'path': os.path.join(base_path, "750ML_Wine/750ML_Wine.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.014387,
    'import_rotations': [0, 0, 0],
    'unconstrained_axis': [0, 0, 1],
    'label_cat': 'n_wine',
    'max_repeats': 2,
}
apple = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "apple/manzana2.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.001365,
    'import_rotations': [0, 0, 0],
    'unconstrained_axis': [0, 0, 1],
    'label_cat': 'n_apple',
    'max_repeats': 3,
}
grapes = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "grapes_1/grapes_1.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.0175,
    'import_rotations': [0, 0, 0],
    'unconstrained_axis': [0, 0, 1],
    'label_cat': 'n_grapes',
    'max_repeats': 2,
}
lettuce = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "LettuceRomaine/LettuceRomaine.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.008,
    'import_rotations': [0, 0, 0],
    'unconstrained_axis': [0, 0, 1],
    'label_cat': 'n_leafy_vegetable',
    'max_repeats': 2,
}
tomato = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "tomato/Tomato_v1.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.009365,
    'import_rotations': [0, 0, 0],
    'unconstrained_axis': [0, 0, 1],
    'label_cat': 'n_tomato',
    'max_repeats': 3,
}
beer_gambrinus = {
    'shelves': [1],
    'path': os.path.join(base_path, "beer_gambrinus/gambrinus.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.02233,
    'import_rotations': [pi/2, 0, 0],
    'unconstrained_axis': [0, 0, 1],
    'label_cat': 'n_bottle_beer',
    'max_repeats': 3,
}

#Combine into one dict
dict_temp = {
    '750ML_Wine': wine,
    'manzana2': apple,
    'grapes_1': grapes,
    'LettuceRomaine': lettuce,
    'Tomato_v1': tomato,
    'gambrinus': beer_gambrinus
}

with open("object_dict.json","w") as f:
    json.dump(dict_temp,f)