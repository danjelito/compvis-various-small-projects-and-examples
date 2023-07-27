import cv2
import matplotlib.pyplot as plt
import argparse


parser= argparse.ArgumentParser()
parser.add_argument("--mode", 
                  type= str, 
                  default= 'image')
parser.add_argument("--file_path", 
                  type= str, 
                  default= 'dataset/celeb faces/Brad Pitt/001_c04300ef.jpg')
args = parser.parse_args()

def detect_face(img, face_detection_object):

    # convert image to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face -> will return x, y, w, h
    face = face_classifier.detectMultiScale(
        image=img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    (x, y, w, h) = face[0]
    return x, y, w, h

def plot_blurred_face(img, x, y, w, h):

    # blur face
    img[y : y + h, x : x + w] = cv2.blur(
        src=img[y : y + h, x : x + w],
        ksize=(50, 50),
    )
    cv2.imshow('press any key to exit', img)
    cv2.waitKey(0)  

# create face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detect face
if args.mode in ['image']:
    
    # read image
    img_path = args.file_path
    img = cv2.imread(img_path)
    x, y, w, h= detect_face(img, face_classifier)
    plot_blurred_face(img, x, y, w, h)

else:
    print('error: mode not recognized')
    
