import numpy as np
import cv2 as cv
import time
cap = cv.VideoCapture(0)

while(True):
    timer = cv.getTickCount()
    ret, frame = cap.read()
    fps = cv.getTickFrequency()/(cv.getTickCount() - timer)
    if fps < 120:
        cv.putText(frame, "FPS : " + str(int(fps)), (40, 50), cv.FONT_HERSHEY_SIMPLEX,  2.35, (20,70,50), 1)
    frameS = cv.resize(frame, (1280, 720))
    cv.imshow('frame',frameS)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows