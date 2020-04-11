# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle
from sklearn.svm import SVC
import argparse
import pickle
import numpy as np

# Modificar las siguientes variables al gusto del directorio montado

embed_path = "resources/pickle/embeddings.pickle"
recognizer_path = "resources/pickle/recognizer.pickle"

# load the face embeddings

data = pickle.loads(open(embed_path, "rb").read())

labels = data["names"]

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("Entrenando al modelo...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
np_labels = np.array(labels)
recognizer.fit(data["embeddings"], np_labels)

# write the actual face recognition model to disk
with open(recognizer_path, "wb") as f:
    f.write(pickle.dumps(recognizer))

print("Saliendo..")
