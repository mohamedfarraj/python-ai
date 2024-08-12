import cv2
from playsound import playsound
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')
cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame,1.2,5)
    for (x,y,w,h) in fire:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h , x:x+w]
        print('Fire is detected')
        playsound('audio.mp3')
    cv2.imshow('Rakwan', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break