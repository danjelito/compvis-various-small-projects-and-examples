import cv2
import mediapipe as mp

import numpy as np
import matplotlib.pyplot as plt
import utils
from pathlib import Path

mp_hands= mp.solutions.hands
mp_drawing= mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles
hands= mp_hands.Hands(
    static_image_mode= True, 
    min_detection_confidence= 0.3,
)

# read images and labels
data_dir= Path('dataset/american sign language digit')
imgs, labels= utils.load_image_dataset(data_dir, True)

# store coordinate per image for all images
data= []

for img in imgs:

    # store coordinate per image
    data_aux= []

    # convert to cv2 image
    img_bgr= cv2.imread(str(img))
    # convert to RGB
    img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 

    results= hands.process(img_rgb)

    # if a hand is predicted
    if results.multi_hand_landmarks: 
        
        # iterate through each landmark
        for hand_landmarks in results.multi_hand_landmarks:
        
            # draw landmarks
            utils.draw_landmarks(mp_hands, 
                                 mp_drawing, 
                                 mp_drawing_styles, 
                                 img_rgb, 
                                 hand_landmarks)
            
            # save coordinate, append to data aux
            for i in range(len(hand_landmarks.landmark)):
                x= hand_landmarks.landmark[i].x
                y= hand_landmarks.landmark[i].y
                z= hand_landmarks.landmark[i].z
                data_aux.append(x)
                data_aux.append(y)
                # data_aux.append(z)
        
        data.append(data_aux)
    
    # # if there is no hand, append 0
    # else:
    #     data.append([np.nan, np.nan, np.nan])   

data= np.array(data)
labels= np.array(labels)
print(data.shape)
print(labels.shape)


# print(sorted(set(labels)))

# plt.imshow(images_rgb[0])
# plt.show()
