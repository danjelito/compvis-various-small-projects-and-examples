import cv2
import mediapipe as mp

import numpy as np
import matplotlib.pyplot as plt

import utils

from pathlib import Path
import pickle

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE, 
    num_hands= 1,)
landmarker= HandLandmarker.create_from_options(options)

# read images and labels
data_dir= Path('dataset/american sign language digit')
imgs, labels= utils.load_image_dataset(data_dir, True)

# store coordinate per image for all images
features= []

# iterate through each image
for img in imgs:

    # store coordinate per image
    img_coor= []

    # convert to cv2 image
    img_bgr= cv2.imread(str(img))
    # convert to RGB
    img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 
    # convert to mp_image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=img_rgb)

    # detect landmark
    results = landmarker.detect(mp_image)
    
    # if a hand is detected:
    if results.hand_landmarks:
        # get hand landmark
        hand_landmarks= results.hand_landmarks[0]
        
        # get 3 coordinate for all 21 landmark
        for landmark in hand_landmarks:
            x= landmark.x
            y= landmark.y
            
            # extend data aux
            img_coor.extend([x, y])

    # if no hand is detected, return all 0
    else:
        img_coor.extend([0] * 42)

    # append this image features to data
    features.append(img_coor)

# convert features and labels to numpy array
features= np.array(features)
labels= np.array(labels)

# save dataset
path= 'dataset/american sign language digit/sign_lang_dataset.pickle'
f= open(file= path, mode= 'wb')
pickle.dump(obj= {'features': features, 'labels': labels}, 
            file= f)
f.close()