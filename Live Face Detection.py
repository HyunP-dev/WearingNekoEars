import cv2
import numpy as np
from toolkit.detection import detect_face

ESC_KEY = 27

video_capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascades\haarcascade_frontalface_alt2.xml")

while True:
    ret, frame = video_capture.read()
    frame = np.array(np.fliplr(frame))
    detections = detect_face(frame)
    for (x1, y1), (x2, y2) in detections:
        cv2.putText(frame, "Face Detected", (x1, y1-2), color=(255, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)
    if k == ESC_KEY:
        break

video_capture.release()
cv2.destroyAllWindows()
