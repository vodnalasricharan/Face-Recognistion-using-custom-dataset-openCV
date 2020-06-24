import cv2
import numpy as np
import os
test_img=cv2.imread('test.jpg')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
gray= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
for (x, y, w, h) in faces:
    print(x, y, w, h)
    # roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
    # roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=4)
resize=cv2.resize(test_img,(550,550))
cv2.imshow('image',resize)
if cv2.waitKey() & 0xFF == ord('q'):
    pass
cv2.destroyAllWindows()