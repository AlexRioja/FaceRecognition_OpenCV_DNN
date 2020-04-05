import cv2                  
import numpy as np         
import sys
import os
import argparse

# argparse para traer el valor de los parametros de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--label", required=True,
	help="Etiqueta que poner a las caras recogidas por el script")
args = vars(ap.parse_args())

face_classifier = cv2.CascadeClassifier('resources/classifiers/haarcascade_frontalface_alt2.xml')


try: 
    os.mkdir("resources/faces_2_recognize/"+args['label']) 
except OSError as error: 
    pass


video_interface = cv2.VideoCapture(0)
n=0
#vamos a crear datasets de por ejemplo 50 fotos :D


while n<50:
	# Leemos el video frame a frame
	ret, frame = video_interface.read()
	#Lo pasamos a escala de grises
	if ret:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#Detectamos las caras
		faces = face_classifier.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(50, 50)
		)

		for (x, y, w, h) in faces:
			#frame=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 150), 0)
			w_rn= int(-0.1*w/2)
			roi_color = frame[y:y+h, x:x+w]
			font = cv2.FONT_HERSHEY_SIMPLEX
			
			cv2.putText(frame, "Carita detectada :)",(5,frame.shape[0]-5),cv2.FONT_HERSHEY_PLAIN, 1.3, (66,53,243), 2, cv2.LINE_AA)

			processed_img=frame
			if processed_img.shape < (300,300):
				processed_img=cv2.resize(processed_img, (300,300),interpolation=cv2.INTER_AREA)
			else:
				processed_img=cv2.resize(processed_img, (300,300),interpolation=cv2.INTER_CUBIC)

			cv2.imwrite("resources/faces_2_recognize/"+args['label']+"/"+str(n)+".jpg", processed_img)

			cv2.waitKey(400) #Capturamos una cara cada X ms
			cv2.imshow('Caritas detectadas', roi_color)
			n+=1

		cv2.imshow('Video WebCam', frame) 
		#Rompemos si pretamos 'q'
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
# Liberamos la interfaz de video y destruimos las ventanas creadas
print('Saliendo...')
video_interface.release()
cv2.destroyAllWindows()
