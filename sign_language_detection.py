import cv2
import mediapipe as mp

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
data_dir= Path('dataset/american sign language')
imgs, labels= utils.load_image_dataset(data_dir, True)

for img in imgs:
    # convert to cv2 image
    img_bgr= cv2.imread(str(img))
    # convert to RGB
    img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 

    results= hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(), 
                mp_drawing_styles.get_default_hand_connections_style(), 
            )

        # print(results.multi_hand_landmarks)
        plt.imshow(img_rgb)
        plt.show()
        break
                

# print(sorted(set(labels)))

# plt.imshow(images_rgb[0])
# plt.show()
