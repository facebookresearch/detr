##used to generate json for object_dict
import os
import json

os.chdir(r"C:\Users\nickh\Documents\GitHub\Fridge-Food-Type\Blender\blender_automation\src\BlenderAPI")
base_path = "C:/Users/nickh/Documents/GitHub/Fridge-Food-Type/Blender/blender_automation/workspace/objects/"

#Individual item dicts, add new models in here
wine = {
    'shelves': [1],
    'path': os.path.join(base_path, "750ML_Wine/750ML_Wine.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.014387,
}
apple = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "apple/manzana2.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.001365,
}
grapes = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "grapes_1/grapes_1.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.0175,
}
lettuce = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "LettuceRomaine/LettuceRomaine.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.008,
}
tomatos = {
    'shelves': [1, 2, 3],
    'path': os.path.join(base_path, "tomato/Tomato_v1.obj"),
    'origin': 'CENTER',
    'scale_factor': 0.009365,
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