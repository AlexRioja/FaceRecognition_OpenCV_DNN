
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from PIL import Image

conf=0.5

#Modificar las siguientes variables al gusto del directorio montado
dataset_path="resources/faces_2_recognize"
embed_path="resources/pickle/embeddings.pickle"
embed_model="resources/model/openface_nn4.small2.v1.t7"
model_path="resources/model/res10_300x300_ssd_iter_140000.caffemodel"
model_proto_path="resources/model/deploy.prototxt"
face_detection_model_path="resources/model"

print("Cargando el sistema...")
detector = cv2.dnn.readNetFromCaffe(model_proto_path, model_path) #carga del detector de caras

embedder = cv2.dnn.readNetFromTorch(embed_model) #carga del modelo embedder

embeddings = []
labels = []

label_ids={}
current_id=0
#Ponemos las label en formato numero en lugar de cadenas de string
def create_ids_4_labels(label):
    global current_id
    id_=0
    if not label in label_ids:
        label_ids[label] = current_id
        current_id +=1
    id_=label_ids[label]
    return id_

for root, dirs, files in os.walk(dataset_path):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path=os.path.join(root, file)
			label=os.path.basename(root).replace(" ", "_").lower()
			print("Procesando: "+ file +" con etiqueta : "+label)
			image = cv2.imread(path)
			image = imutils.resize(image, width=600)
			(h, w) = image.shape[:2]
			# construimos el blob de la imagen
			imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

			#localizamos las caras en la imagen
			detector.setInput(imageBlob)
			caras = detector.forward()

			if len(caras) > 0:
				count = np.argmax(caras[0, 0, :, 2])
				confidence = caras[0, 0, count, 2]
				print(confidence)
				#Comprobamos que el nivel de confianza es mayor al que le hemos puesto como limite inferior
				if confidence>conf:
					box = caras[0, 0, count, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					face = image[startY:endY, startX:endX]

					
					# construct a blob for the face ROI, then pass the blob
					# through our face embedding model to obtain the 128-d
					# quantification of the face
					try:
						cv2.imshow('Face',face)
						cv2.waitKey(40)
						if cv2.waitKey(1) & 0xFF == ord('q'):
							break
						faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
							(96, 96), (0, 0, 0), swapRB=True, crop=False)
						embedder.setInput(faceBlob)
						vec = embedder.forward()
						
						labels.append(create_ids_4_labels(label))
						embeddings.append(vec.flatten())
					except:
						print("Cara no centrada en la imagen. Frame corrupto")
					
print(label_ids)
print(labels)
print(len(labels))
data = {"embeddings": embeddings, "names": labels}
with open(embed_path, "wb") as f:
	f.write(pickle.dumps(data))
	
with open("resources/pickle/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
print("Saliendo..")