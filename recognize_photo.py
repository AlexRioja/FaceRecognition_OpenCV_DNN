import numpy as np
import imutils
import pickle
import cv2
"""Rutas a los archivos necesarios
"""
# Modelo que pasa de una foto de cara a un vector de 128D con la info de la cara
embed_model = "resources/model/openface_nn4.small2.v1.t7"
# Modelo SVC que se encarga de obtener la recta que separa las regiones de caras en el espacio
recognizer_path = "resources/pickle/recognizer.pickle"
# Modelo preentrenado para la detección de rostros en una imagen
model_proto_path = "resources/model/deploy.prototxt"
model_path = "resources/model/res10_300x300_ssd_iter_140000.caffemodel"
"""Cargamos el detector de rostros y el embedder
"""
detector = cv2.dnn.readNetFromCaffe(model_proto_path, model_path)
embedder = cv2.dnn.readNetFromTorch(embed_model)
"""Abrimos el archivo donde hemos guardado a qué cara se corresponde qué vector de datos
    Es decir, las etiquetas
"""
with open("resources/pickle/labels.pickle", "rb") as f:
    inv_labels = pickle.load(f)
    labels = {v: k for k, v in inv_labels.items()}
recognizer = pickle.loads(open(recognizer_path, "rb").read())

# Cargamos y pre-procesamos la imagen, obtenemos sus dimensiones
image = cv2.imread("res/1.jpg")
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
# obtenemos el blob de la imagen-->representación abstracta
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)
# usamos el detector de caras de OpenCV para detectarlas
detector.setInput(imageBlob)
detections = detector.forward()

# loopeamos las caras
for i in range(0, detections.shape[2]):
    # sacamos la confianza (probabilidad) de cada predicción
    confidence = detections[0, 0, i, 2]
    # si supera nuestro Umbral de confianza
    if confidence > 0.5:
        # coordenadas de la cara detectada
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # recortamos de la imagen la cara detectada
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
    try:
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()  # contiene la representacion de las 128D de la cara

    except:
        print("Cara corrupta o no centrada en el campo de vision de cámara. Frame corrupto")

    # llamamos a nuestro modelo para que nos de el vector de predicciones
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = labels[j]
    # dibujamos encima de la foto el rectangulo y la etiqueta
    text = "{}: {:.2f}%".format(name, proba * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
    cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# mostramos la imagen de salida
cv2.imshow("Prueba reconocimiento sobre imagenes", image)
while True:
    key_pressed=cv2.waitKey(200)
    if cv2.getWindowProperty('Prueba reconocimiento sobre imagenes',cv2.WND_PROP_VISIBLE) < 1:        
        break
cv2.destroyAllWindows()  

