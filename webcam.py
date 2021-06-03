import cv2
import numpy as np

ESC_KEY = 27

video_capture = cv2.VideoCapture(0)


while True:
    ret, frame = video_capture.read()
    frame = np.array(np.fliplr(frame))
    
    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)
    if k == ESC_KEY:
        break

video_capture.release()
cv2.destroyAllWindows()
