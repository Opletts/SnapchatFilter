import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while(True):
	_, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	eye_cascade = cv2.CascadeClassifier('/opt/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')

	eyes = eye_cascade.detectMultiScale(gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow("Eyes", img)
	cv2.waitKey(1)