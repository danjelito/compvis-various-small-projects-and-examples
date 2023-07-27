import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="video")
parser.add_argument(
    "--file_path", type=str, default="dataset/person talking/person-talking-1.mp4"
)
args = parser.parse_args()


def detect_face(img, face_detection_object):
    # convert image to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face -> will return x, y, w, h
    face = face_detection_object.detectMultiScale(
        image=img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    # if face is not detected, return all 0
    if len(face) == 0:
        return 0, 0, 0, 0
    # if face is detected, return bbox
    (x, y, w, h) = face[0]
    return x, y, w, h


def blur_face(img, x, y, w, h):
    
    # if there is no face, return original img
    if all(obj == 0 for obj in [x, y, w, h]):
        return img
    # if there is face, blur
    img[y : y + h, x : x + w] = cv2.blur(
        src=img[y : y + h, x : x + w],
        ksize=(50, 50),
    )
    return img


def plot_blurred_face(img):
    cv2.imshow("press any key to exit", img)
    cv2.waitKey(0)


# create face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detect face
if args.mode in ["image"]:
    # read image
    img_path = args.file_path
    img = cv2.imread(img_path)
    # get face bbox
    x, y, w, h = detect_face(img, face_classifier)
    # blur face
    img = blur_face(img, x, y, w, h)
    # display
    plot_blurred_face(img)

elif args.mode in ["video"]:
    # read video
    cap = cv2.VideoCapture(args.file_path)
    ret, frame = cap.read()

    # create video object
    original_filename = str(Path(args.file_path).stem)
    output_filepath = str(Path(f"output/video/{original_filename}_blurred.mp4"))
    output_video = cv2.VideoWriter(
        filename=output_filepath,
        fourcc=cv2.VideoWriter_fourcc(*"MP4V"),
        fps=25,
        frameSize=(frame.shape[1], frame.shape[0]),
    )

    while ret:
        # get face bbox
        x, y, w, h = detect_face(frame, face_classifier)
        # if face is found, blur face
        frame = blur_face(frame, x, y, w, h)
        # write video
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()

elif args.mode in ["debug"]:
    # read image
    img_path = args.file_path
    img = cv2.imread(img_path)
    # get face bbox
    detect_face(img, face_classifier)

else:
    print("error: mode not recognized")
