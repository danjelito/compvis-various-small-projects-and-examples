import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import utils

# load mediapipe options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE, 
    num_hands= 1,)
landmarker= HandLandmarker.create_from_options(options)

# load model
path= 'model/sign_digit.pickle'
clf= pickle.load(file= open(path, 'rb'))
clf= clf['model']

# capture video
cap= cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()

    # convert to mp_image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=frame)

    # detect landmark
    results = landmarker.detect(mp_image)
    
    # if a hand is detected:
    if results.hand_landmarks:
        
        # get hand landmark
        hand_landmarks= results.hand_landmarks[0]
        
        # store coordinate per frame
        img_coor= []

        # get 3 coordinate for all 21 landmark
        for landmark in hand_landmarks:
            x= landmark.x
            y= landmark.y
            
            # extend img_coor
            img_coor.extend([x, y])

        # predict
        features= np.array(img_coor).reshape(1, -1)
        y_pred= clf.predict(features)
        print(y_pred)

        # draw landmark
        frame = utils.draw_landmarks_on_image(
            frame, 
            results
        )
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()