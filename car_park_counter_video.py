import pickle
import pandas as pd
import numpy as np
from skimage.transform import resize
import cv2

def get_parking_spot_bboxes(connected_components):
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

# load video
video_path= 'dataset/parking lot/video/parking_1920_1080_loop.mp4'
cap= cv2.VideoCapture(video_path)

# load mask
mask_path= 'dataset/parking lot/mask/mask_1920_1080.png'
mask= cv2.imread(mask_path, flags= 0)
connected_comp= cv2.connectedComponentsWithStats(image= mask, 
                                                 labels= 4, 
                                                 connectivity= cv2.CV_32S)
spots= get_parking_spot_bboxes(connected_comp)

ret= True
while ret:
    ret, frame= cap.read()

    # draw parking spots
    for spot in spots:
        x1, y1, w, h= spot
        cv2.rectangle(img= frame, 
                      pt1= (x1, y1), 
                      pt2= (x1+w, y1+h), 
                      color= (255, 0, 0), 
                      thickness= 2)
    
    # visualize
    window_title= 'press q to quit'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1000, 750)
    cv2.imshow(window_title, frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()