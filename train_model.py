from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
from joblib import dump, load
import numpy as np

model = Sequential([Dense(128, input_shape = (128,), activation = 'relu'),
                    Dense(64, activation = 'relu', kernel_initializer = 'he_uniform'),
                    Dense(64, activation = 'relu',  kernel_initializer = 'he_uniform'),
                    Dense(32, activation = 'relu',  kernel_initializer = 'he_uniform'),
                    Dense(32, activation = 'relu',  kernel_initializer = 'he_uniform'),
                    Dense(23, activation = 'softmax')])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam' ,metrics=['accuracy'])

# load the face embeddings
print("[INFO] loading face embeddings...")
#data = load(open("output/embeddings.joblib", "rb").read())
data = load("output/embeddings.joblib")

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

embeddings = data["embeddings"]
embeddings = np.array(embeddings)
h = model.fit(embeddings, labels, epochs = 50)


model.save('recognizer.h5')
with open('output/le.joblib', 'wb') as f:  
    dump(le, f)
f.close()
'''
# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
#f = open("output/recognizer", "wb")
#f.write(dump(recognizer,"output/recognizer"))
with open('output/recognizer.joblib', 'wb') as f:  
    dump(recognizer, f)

f.close()

# write the label encoder to disk
#f = open("output/le.joblib", "wb")
#f.write(dump(le ,"output/le.joblib"))
'''
