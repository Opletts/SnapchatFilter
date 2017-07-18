import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while(True):
	_, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	eye_cascade = cv2.CascadeClassifier('/opt/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml')

	eyes = eye_cascade.detectMultiScale(gray)
	
	if len(eyes) == 2:

		x = min(eyes[0][0], eyes[1][0])
		if x == eyes[0][0]:
			w = eyes[1][0]+eyes[1][2]-eyes[0][0]
		else:
			w = eyes[0][0]+eyes[0][2]-eyes[1][0]

		y = min(eyes[0][1], eyes[1][1])
		h = max(eyes[0][3], eyes[1][3])

		gl = cv2.imread('glass.jpg')
		#cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
		gl = cv2.resize(gl, (w+50,h+50))

		roi = img[y-25:y+h+25, x-25:x+w+25]

		glgray = cv2.cvtColor(gl,cv2.COLOR_BGR2GRAY)
		_, mask = cv2.threshold(glgray, 10, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
		fg = cv2.bitwise_and(gl,gl,mask = mask)
		dst = cv2.add(bg, fg)
		img[y-25:y+h+25, x-25:x+w+25] = dst

		cv2.imshow("Eyes", img)
		cv2.waitKey(1)
