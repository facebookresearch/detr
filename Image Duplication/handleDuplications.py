import pandas as pd
import numpy as np
import cv2
import glob
import os
import argparse

def getHashes(imagePaths, args):
    hashes = {}
    # loop over our image paths
    for i in range(len(imagePaths)):
        # load the input image and compute the hash
        try:
            image = cv2.imread(imagePaths[i])
           
        except:
            print("image", imagePaths[i], " cant be opened and will be deleted")
            os.remove(imagePaths[i])
            continue
            
        if(image is not None):
            #collect minimum dimension for size filtering on dataset
            min_image_dim = min(np.shape(image)[0], np.shape(image)[1])
            #Image removed from directory if either h or w < min_size argument
            #Done in gethashes function as it loops through all images already
            if min_image_dim < args.min_size:
                print("image", imagePaths[i], " is too small and will be deleted")
                os.remove(imagePaths[i])
            else:
                
                h = dhash(image)
                # grab all image paths with that hash, add the current image
                # path to it, and store the list back in the hashes dictionary
                p = hashes.get(h, [])
                p.append(imagePaths[i])
                hashes[h] = p
        else:
            #final catch for reading in image succesfully, but as null
            print("image cannot be opened --deleted from paths")
            os.remove(imagePaths[i])
        
    return hashes


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    #this simpy detects the edges -> returns True if adjacent column has bigger values    
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def countDuplications(hashes):
    count = 0
    for key, value in hashes.items():
        if(len(value)>1):
            count += 1
            
    return count

def main(args):
    imagePaths = np.array(sorted(glob.glob(os.path.join(args.directory, "*"))))
    print("Images to be processed: ", len(imagePaths))
    hashes = getHashes(imagePaths, args)    
    
    print("number of duplicated images:" ,  countDuplications(hashes))
    handleDuplicates(hashes)
    print('Finished... :)')

def handleDuplicates(hashes):
    for (h, hashedPaths) in hashes.items():
        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:
            # loop over all image paths with the same hash *except*
            # for the first image in the list (since we want to keep
            # one, and only one, of the duplicate images)
            for p in hashedPaths[1:]:
                os.remove(p)


def argument_parser():
    parser = argparse.ArgumentParser(description='Delete Duplicate Images')
    parser.add_argument('-D', '--directory', default='./fridge', type=str, help='directory within which to delete repeated images')
    parser.add_argument('-M', '--min_size', default=200, type=int, help='minimum image size- single dimension. Images will be deleted if either h or w smaller than argument.')
    args = parser.parse_args()
    
    if not os.path.isdir(os.path.abspath(args.directory)):
        raise Exception("Given directory : {}\n doesn\'t exists, pass \'--create_directory\' flag or a valid directory".format(os.path.abspath(args.directory)))	
    args.directory = os.path.abspath(args.directory)
    return args

if __name__ == '__main__':
    main(argument_parser())
