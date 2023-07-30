import numpy as np
import cv2
import os

def get_color_limits(color):

    c= np.uint8([[color]]) 
    hsv_c= cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lower_limit= hsv_c[0][0][0] - 10, 100, 100
    upper_limit= hsv_c[0][0][0] + 10, 255, 255
    
    lower_limit= np.array(lower_limit, dtype= np.uint8)
    upper_limit= np.array(upper_limit, dtype= np.uint8)

    return lower_limit, upper_limit

def load_image_dataset(dataset_dir, shuffle= False):

    images_paths= []
    labels= []

    # iterate through each image folder
    for c in os.listdir(dataset_dir):
        c_dir = dataset_dir / c
        # if not a folder, continue
        if os.path.isfile(c_dir):
            continue
        # list all images inside image folder
        images= os.listdir(c_dir)
        # append to list
        for image in images:
	        # if in image format, append to list
            image_formats= ['jpg', 'jpeg', 'png']
            if any(image.endswith(ext) for ext in image_formats):
               image_path= c_dir / image 
               images_paths.append(image_path)
               labels.append(c)

    # shuffle dataset
    if shuffle:
        import random
        random.seed(0)
        indices = list(zip(images_paths, labels))
        random.shuffle(indices)
        images_paths, labels = zip(*indices)

    return images_paths, labels

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image