import cv2
import numpy as np
import dlib
from toolkit.transformation import rotate_image, paste_image
from toolkit.measurement import get_rotation
from math import sin, radians
ESC_KEY = 27

video_capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "landmarks\shape_predictor_68_face_landmarks.dat")

ears_orig = cv2.imread(r"dataset\neko_ears\6570766_preview.png")


def get_euclidean_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5


def draw_landmark_number(photo, landmark):
    result = photo.copy()
    for i in range(landmark.num_parts):
        x = landmark.part(i).x
        y = landmark.part(i).y

        cv2.putText(result, str(i), (x, y),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
    return result


def determine_ears_size(landmark):
    p1 = (landmark.part(0).x, landmark.part(0).y)
    p2 = (landmark.part(16).x, landmark.part(16).y)
    distance = int(get_euclidean_distance(p1, p2))

    ratio = distance / ears.shape[1]
    dim = (distance, int(ears.shape[0] * ratio))
    return dim


def determine_ears_location(landmark, ears, rotation):
    p1 = (landmark.part(0).x, landmark.part(0).y)
    p2 = (landmark.part(16).x, landmark.part(16).y)

    if rot >=0:
        loc_x = p1[0]-int(ears.shape[1]*sin(radians(rotation)))
    else:
        loc_x = p1[0]-int(0.5*ears.shape[1]*sin(radians(rotation)))

    if rot >= 0:
        loc_y = p1[1]-ears.shape[1]
    else:
        loc_y = p2[1]-ears.shape[1]

    return (loc_x, loc_y)


while True:
    ret, frame = video_capture.read()
    frame = np.array(np.fliplr(frame))
    frame_orig = frame.copy()
    try:
        faces = detector(frame)
        landmark = predictor(frame, faces[0])
        # frame = draw_landmark_number(frame, landmark)

        ears = ears_orig.copy()
        resized = cv2.resize(ears, determine_ears_size(landmark),
                             interpolation=cv2.INTER_NEAREST)

        rot = get_rotation(frame)
        print("rotation:", rot)
        ears = rotate_image(resized, rot)

        cv2.imshow('Video', paste_image(frame, ears, determine_ears_location(landmark, ears, rot)))
    except Exception as e:
        print("ERROR")
        cv2.imshow('Video', frame)

    k = cv2.waitKey(1)
    if k == ESC_KEY:
        break

video_capture.release()
cv2.destroyAllWindows()
