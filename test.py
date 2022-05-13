# For the use of the GPU with tensorflow
import os
from pandas import Categorical
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

from tensorflow import keras
from dataPreprocessing import dataPreprocessing

#Load model and weights
with open("model/model.json", "r") as json_file:
    model_json = json_file.read()

model = keras.models.model_from_json(model_json)
model.load_weights("model/model.h5")

# Prepare test set 30%
x_test, y_test = dataPreprocessing(58000, None)

predictions = (model.predict(x_test))

acc = keras.metrics.CategoricalAccuracy()
acc.update_state(y_test, predictions)
accuracy = acc.result().numpy()

cce = keras.metrics.CategoricalCrossentropy()
cce.update_state(y_test, predictions)
entropy = cce.result().numpy()

pre = keras.metrics.Precision()
pre.update_state(y_test, predictions)
precision = pre.result().numpy()

rec = keras.metrics.Recall()
rec.update_state(y_test, predictions)
recall = rec.result().numpy()

print("Accuracy : ", accuracy)
print("Categorical Crossentropy : ", entropy)
print("Precision : ", precision)
print("Recall : ", recall)