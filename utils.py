import numpy as np
import cv2
import os

def get_color_limits(color):

    c= np.uint8([[color]]) 
    hsv_c= cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lower_limit= hsv_c[0][0][0] - 10, 100, 100
    upper_limit= hsv_c[0][0][0] + 10, 255, 255
    
    lower_limit= np.array(lower_limit, dtype= np.uint8)
    upper_limit= np.array(upper_limit, dtype= np.uint8)

    return lower_limit, upper_limit

def load_image_dataset(dataset_dir, shuffle= False):

    images_paths= []
    labels= []

    # iterate through each image folder
    for c in os.listdir(dataset_dir):
        c_dir = dataset_dir / c
        # if not a folder, continue
        if os.path.isfile(c_dir):
            continue
        # list all images inside image folder
        images= os.listdir(c_dir)
        # append to list
        for image in images:
	        # if in image format, append to list
            image_formats= ['jpg', 'jpeg', 'png']
            if any(image.endswith(ext) for ext in image_formats):
               image_path= c_dir / image 
               images_paths.append(image_path)
               labels.append(c)

    # shuffle dataset
    if shuffle:
        import random
        random.seed(0)
        indices = list(zip(images_paths, labels))
        random.shuffle(indices)
        images_paths, labels = zip(*indices)

    return images_paths, labels