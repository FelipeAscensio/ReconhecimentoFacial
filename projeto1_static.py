#importa as bibliotecas nescessaria para a utilização da webcam ou quadro estatico
import numpy as np
import cv2

#esse arquivo .xml é conjunto de dados para reconhecimento facial baseado em  arrays e matrizes multidimensionais usado pelo numpy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('C:\\foto.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
   cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow('img',img)

cv2.waitKey()
