import cv2
import easyocr
import matplotlib.pyplot as plt

# read image
img_path= 'dataset/signboard/5.jpg'
img= cv2.imread(img_path)

# create text detector instance
reader= easyocr.Reader(['en'], gpu= False)

# detect text
texts= reader.readtext(img)

# draw bbox and text
threshold= 0.25
for t in texts:
    bbox, text, score= t

    # convert bbox to int
    bbox_int = [[int(round(element)) for element in inner_list] for inner_list in bbox]
    
    if score  > threshold:
        cv2.rectangle(img= img, 
                      pt1= bbox_int[0], 
                      pt2= bbox_int[2], 
                      color= (0, 255, 0), 
                      thickness= 5)
        cv2.putText(img= img, 
                    text= text, 
                    org= bbox_int[0], 
                    fontFace= cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale= 0.75, 
                    color= (255, 0, 0), 
                    thickness= 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
