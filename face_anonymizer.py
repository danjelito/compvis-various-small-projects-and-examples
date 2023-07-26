import cv2
import matplotlib.pyplot as plt

def detect_face(img, face_detection_object):

    # convert image to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face -> will return x, y, w, h
    face = face_classifier.detectMultiScale(
        image=img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    (x, y, w, h) = face[0]
    return x, y, w, h

# read image
img_path = "dataset/celeb faces/Brad Pitt/001_c04300ef.jpg"
img = cv2.imread(img_path)

# create face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detect face
x, y, w, h= detect_face(img, face_classifier)

# plot face with bbox
plot = False
if plot:
    cv2.rectangle(
        img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=4
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img_rgb)
    cv2.waitKey(0)

# blur face
img[y : y + h, x : x + w] = cv2.blur(
    src=img[y : y + h, x : x + w],
    ksize=(50, 50),
)
cv2.imshow("image", img)
cv2.waitKey(0)