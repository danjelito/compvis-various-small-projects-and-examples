import pickle
import pandas as pd
import numpy as np
from skimage.transform import resize
import cv2
from datetime import datetime

def get_parking_spot_bboxes(connected_components):
    """Return the coordinate of boox from connected components"""
    (total_label, label_ids, values, centroids)= connected_components
    slots= []
    coef= 1
    for i in range(1, total_label):
        # extract coordicate
        x1= int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1= int(values[i, cv2.CC_STAT_TOP] * coef)
        w= int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h= int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots

def is_empty(clf, spot_img_bgr):
    """Predict if spot is empty or not"""
    img= resize(spot_img_bgr, (15, 15, 3)).flatten().reshape(1, -1)
    y_pred= clf.predict(img)
    return y_pred


# load video
video_path= 'dataset/parking lot/video/parking_1920_1080_loop.mp4'
cap= cv2.VideoCapture(video_path)

# load mask
mask_path= 'dataset/parking lot/mask/mask_1920_1080_better.png'
mask= cv2.imread(mask_path, flags= 0)
connected_comp= cv2.connectedComponentsWithStats(image= mask, 
                                                 labels= 4, 
                                                 connectivity= cv2.CV_32S)
spots= get_parking_spot_bboxes(connected_comp)

# load clf model
model_path= 'model/car_park_clf.pickle'
clf= pickle.load(open(model_path, 'rb'))

frame_per_clf= 150  # predict every 30 frames
frame_number= 0

# list of all spots' status
spot_statuses= [None for i in spots]

ret= True
while ret:
    ret, frame= cap.read()

    # get each parking spot to predict
    for spot_idx, spot in enumerate(spots):
        x1, y1, w, h= spot

        if frame_number % frame_per_clf == 0:
            spot_crop= frame[y1:y1+h, x1:x1+w, :]
            spot_status= is_empty(clf, spot_crop)
            spot_statuses[spot_idx] = spot_status

    # draw parking spots
    # red if not empty, else green
    for spot, spot_status in zip(spots, spot_statuses):
        x1, y1, w, h= spot

        color= (0, 255, 0) if spot_status == 'empty' else (0, 0, 255) 
        cv2.rectangle(img= frame, 
                    pt1= (x1, y1), 
                    pt2= (x1+w, y1+h), 
                    color= color, 
                    thickness= 2)
    
    # update frame number
    frame_number += 1
    
    # visualize
    window_title= 'press q to quit'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)
    cv2.imshow(window_title, frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows() 