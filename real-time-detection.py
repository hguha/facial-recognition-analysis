import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import numpy as np

faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier( "models/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("models/haarcascade_smile.xml")
video_capture = cv2.VideoCapture(0)

while True:
    
    ret, frame = video_capture.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        # face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame,'Face',(x, y), 1, 2, (255,0,0), 5)
        
        cropped_img = img[y:y+h, x:x+w]
        cropped_img_color = frame[y:y+h, x:x+w]
        
        # mouth
        smiles = smileCascade.detectMultiScale(cropped_img, scaleFactor= 1.16, minNeighbors=40, minSize=(25, 25), flags=cv2.CASCADE_SCALE_IMAGE)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(cropped_img_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)
              
        # eyes
        eyes = eyeCascade.detectMultiScale(cropped_img, minSize=(10, 10), minNeighbors=20)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(cropped_img_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Release Capture
video_capture.release()
cv2.destroyAllWindows()