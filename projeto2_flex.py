#importa as bibliotecas nescessaria para a utilização da webcam ou quadro estatico
import numpy as np
import cv2


#esse arquivo .xml é conjunto de dados para reconhecimento facial baseado em  arrays e matrizes multidimensionais usado pelo numpy
rosto_repo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#inicia a utilização da webcam
webcam = cv2.VideoCapture(0)

#entra em loop para que a instancia da webcam fique operante
while(True):

   s,video = webcam.read()

#deixa o video na posição correta
   video = cv2.flip(video,180)
 
#define o tamanho dos retangulos
   faces = rosto_repo.detectMultiScale(video,minNeighbors = 20, minSize = (30,30), maxSize = (400,400))

#define a cor de todos os lados do retangulo
   for (x,y,w,h) in faces:
     cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),4)

#exibe o resultado 
   cv2.imshow("Rosto detectado", video)

#exibe o video em modo continuo sem nenhuma enterrupção   
   if (cv2.waitKey(1) and 0xFF == ord('q')):
      break
