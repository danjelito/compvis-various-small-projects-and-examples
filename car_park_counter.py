import os
import utils
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np

# get images and labels
image_dir= Path('dataset/parking lot/image')
images, labels= utils.load_image_dataset(image_dir, shuffle= False)

# convert images from path to images
images= [imread(i) for i in images]

# resize image to 15x15
images= [resize(i, (15, 15)) for i in images]

# flatten image to 1-d array
# convert labels to array
images= np.asarray([i.flatten() for i in images])
labels= np.asarray(labels)

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    images, 
    labels,
    stratify= labels,
    test_size= 0.2,
    shuffle= True,  
    random_state= 42
)

print(images.shape, labels.shape)