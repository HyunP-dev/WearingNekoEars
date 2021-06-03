import cv2
import numpy as np
import dlib

ESC_KEY = 27

video_capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("landmarks\shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = video_capture.read()
    frame = np.array(np.fliplr(frame))
    
    faces = detector(frame)
    for face_index, face in enumerate(faces):
        landmark = predictor(frame, face)
        cv2.rectangle(frame, (face.left(), face.top()),
                             (face.right(), face.bottom()), (0, 255, 0))
        for i in range(landmark.num_parts):
            x = landmark.part(i).x
            y = landmark.part(i).y

            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))
    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)
    if k == ESC_KEY:
        break

video_capture.release()
cv2.destroyAllWindows()
