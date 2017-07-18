import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while(True):
	_, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	face_cascade = cv2.CascadeClassifier('/opt/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		ch = cv2.imread('Cherizerd.png') #read the image you want to overlay on the face
		#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		ch = cv2.resize(ch, (w,h))

		roi = img[y:y+h, x:x+w]

		chgray = cv2.cvtColor(ch,cv2.COLOR_BGR2GRAY)
		ret, mask = cv2.threshold(chgray, 10, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
		fg = cv2.bitwise_and(ch,ch,mask = mask)
		dst = cv2.add(bg, fg)
		img[y:y+h, x:x+w] = dst 

	cv2.imshow('img',img)	
	cv2.waitKey(1)