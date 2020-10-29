"""
Used to generate object dictionary for random placement.
shelves: shelves that object is allowed to be placed on (enum in 1,2,3)
path: path to object file within blender_automation/workspace directory
origin: possibly depriciated, where origin of item is stored. Should be CENTER for all .obj files?
scale_factor: scale factor to apply to object on import.
unconstrained axis: axis allowed for euler random rotation. Only 1 axis per item right now.
    Stored as T/F binary list (x, y, z)
"""
import os
import json

os.chdir(r"C:\Users\nickh\Documents\GitHub\Fridge-Food-Type\Blender\blender_automation\src")
base_path = "C:/Users/nickh/Documents/GitHub/Fridge-Food-Type/Blender/blender_automation/workspace/objects/"
#Individual object parameters
wine = {
    'shelves': [1],
    'path': os.path.join(base_path, "750ML_Wine/750ML_Wine.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.014387,
    'unconstrained_axis': [0, 0, 1],
}
apple = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "apple/manzana2.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.001365,
    'unconstrained_axis': [0, 0, 1],
}
grapes = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "grapes_1/grapes_1.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.0175,
    'unconstrained_axis': [0, 0, 1],
}
lettuce = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "LettuceRomaine/LettuceRomaine.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.008,
    'unconstrained_axis': [0, 0, 1],
}
tomatos = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "tomato/Tomato_v1.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.009365,
    'unconstrained_axis': [0, 0, 1],
}

#Combine into one dict
dict_temp = {
    '750ML_Wine': wine,
    'manzana2': apple,
    'grapes_1': grapes,
    'LettuceRomaine': lettuce,
    'Tomato_v1': tomatos
}

with open("object_dict.json","w") as f:
    json.dump(dict_temp,f)