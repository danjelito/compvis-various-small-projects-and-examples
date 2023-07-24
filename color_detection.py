import cv2
import utils
from PIL import Image

cap= cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()

    # convert BGR to HSV
    hsv_image= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get shades of yellow
    yellow= [0, 255, 255] # BGR -> green + red
    lower_limit, upper_limit= utils.get_color_limits(yellow)

    # get mask of selected color
    mask= cv2.inRange(src= hsv_image, 
                      lowerb= lower_limit, 
                      upperb= upper_limit)
    
    # convert numpy to PIL
    mask_= Image.fromarray(mask)

    # get and draw bbox
    bbox= mask_.getbbox()
    if bbox is not None:
        x1, y1, x2, y2= bbox
        cv2.rectangle(img= frame, 
                      pt1= (x1, y1), 
                      pt2= (x2, y2), 
                      color= (255, 0, 0),
                      thickness= 5)

    # show video
    cv2.imshow('frame', frame)

    # break loop if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
