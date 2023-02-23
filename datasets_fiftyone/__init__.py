import fiftyone as fo 

# --- CONFIG --- #
dataset = fo.zoo.load_zoo_dataset("coco-2017",split="validation",  max_samples=50,label_types=["detections"])
#FIXME add params as conf
#session = fo.launch_app(dataset)
