## Blender Fridge Image Creation

The scripts contained in this folder are used to create an image within blender utilizing the Blender-python API. 

##### Requirements:
Blender 2.9 - comes packaged with python 3.77
	
All those packages can be downloaded straight from pip if not installed

### how to use
Open blender, click the scripting tab, then open the main fridge generation file. The file can be run from within Blender to see the results visually. 

Alternatively, it can be run using the command line as well (remove the --background argument if you'd like to see the Blender GUI). 
blender --background --python fridge.py 

##### To-Do: 
1. Automatic placement of items
    * Random shelf selection, coordinate setting
    * Object collision check before movement
2. Model collection for food items
    * Are there better options than the .obj format?
    * Preprocessing all objects so they can be added to scene without resizing/rotation
3. Render Engine Choice finalization 
    * Current choice: Cycles render
    * Other engines don't show clear materials well, need to fix if moving away from Cycles engine

