from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import cv2
import os

# Modificar las siguientes variables al gusto del directorio montado
embed_model = "resources/model/openface_nn4.small2.v1.t7"
recognizer_path = "resources/pickle/recognizer.pickle"
face_detection_model_path = "resources/model"
model_proto_path = "resources/model/deploy.prototxt"
model_path = "resources/model/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(model_proto_path, model_path)
embedder = cv2.dnn.readNetFromTorch(embed_model)
with open("resources/pickle/labels.pickle", "rb") as f:
    inv_labels = pickle.load(f)
    labels = {v: k for k, v in inv_labels.items()}
recognizer = pickle.loads(open(recognizer_path, "rb").read())

video_interface = cv2.VideoCapture(0)
# contador aproximado de FPS
fps = FPS().start()

while True:
    ret, frame = video_interface.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construimos el blob desde la imagen
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300))

    detector.setInput(imageBlob)
    faces = detector.forward()

    # loopeamos las caras que encontremos
    for i in range(0, faces.shape[2]):
        # sacamos la confianza (probabilidad) de cada predicción
        confidence = faces[0, 0, i, 2]

        # filtramos falsos positivos con nivel bajo de confianza
        if confidence > 0.5:
            # coordenadas de la cara detectada
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            try:
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
            except:
                print("Cara no centrada en el campo de vision de cámara. Frame corrupto")

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = labels[j]  # le.classes_[j]

            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    fps.update()
    cv2.putText(frame, "Q para salir", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)
    cv2.imshow("Video-Reconocimiento-Facial", frame)
    key = cv2.waitKey(1) & 0xFF

    # rompemos bucle con la tecla q
    if key == ord("q"):
        break

fps.stop()
print("FPS aproximados: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
video_interface.release()
print("Saliendo..")
