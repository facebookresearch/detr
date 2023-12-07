import glob
import sys
import shutil
from PIL import Image, ImageDraw
import sys
import requests

import matplotlib.pyplot as plt

def add_white_rectangle(img_path, online = False):
    IMAGE_PADDING = 250
    if (online == False):
        img = Image.open(img_path).convert("RGB")
    else:
        img = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
    # Open the image file

    # Create a new image with the same width and increased height
    new_img = Image.new('RGB', (img.width, img.height + IMAGE_PADDING), color='white')

    # Paste the original image onto the new image
    new_img.paste(img, (0, 0))

    # Draw a white rectangle at the bottom of the new image
    draw = ImageDraw.Draw(new_img)
    draw.rectangle((0, img.height, img.width, img.height + IMAGE_PADDING), fill='white')


    # Return the new image
    return new_img

def export_tracking(source_path, dest_path):
    chosen_folder = [13, 16, 17, 19]
    img_cnt = 100000
    for folder in chosen_folder:
        folder_name = str(str(folder)).zfill(4)
        folder_path = source_path + folder_name + "\\"
        print("extracting folder " + folder_name)
        print("folder path: " + folder_path)
        img_path_set = glob.glob(folder_path + "*.png") + glob.glob(folder_path + "*.jpg")
        img_path_set.sort()
        for source_img_path in img_path_set:
            # copy file to destination
            img_name = str(img_cnt).zfill(6) + source_img_path[-4:]
            img_cnt += 1
            dest_img_path = dest_path + img_name
            print("copying " + source_img_path + " to " + dest_img_path)
            # shutil.copyfile(source_img_path, dest_img_path)
            padded_img = add_white_rectangle(source_img_path, False)
            padded_img.save(dest_img_path)

def export_detection(source_path, dest_path):
    img_cnt = 0
    img_path_set = glob.glob(source_path + "*.png") + glob.glob(source_path + "*.jpg")
    img_path_set.sort()
    for source_img_path in img_path_set:
        # copy file to destination
        img_name = str(img_cnt).zfill(6) + source_img_path[-4:]
        img_cnt += 1
        dest_img_path = dest_path + img_name
        print("copying " + source_img_path + " to " + dest_img_path)
        # shutil.copyfile(source_img_path, dest_img_path)

def export(source_path, dest_path, type):
    if type == "Tracking":
        export_tracking(source_path, dest_path)
    elif type == "Detection":
        export_detection(source_path, dest_path)
    else:
        print("Wrong type")

for arg in sys.argv:
    print(arg, len(arg))

if len(sys.argv) != 4:
    print("Usage: python merge_tracking_object.py <source_path> <dest_path> <type>")
    sys.exit(1)

source_path = sys.argv[1]
dest_path = sys.argv[2]
type = sys.argv[3]

if source_path[-1] != "\\":
    source_path += "\\"

if dest_path[-1] != "\\":
    dest_path += "\\"

export(source_path, dest_path, type)