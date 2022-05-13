# For the use of the GPU with tensorflow
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

from tensorflow import keras
from dataPreprocessing import dataPreprocessing

#Load model and weights
with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = keras.models.model_from_json(model_json)
model.load_weights("model.h5")

# Prepare test set 30%
x_test, y_test = dataPreprocessing(58000, None)

predictions = (model.predict(x_test)).round()
print(predictions, y_test)

acc = keras.metrics.BinaryAccuracy(threshold = 0.5)
acc.update_state(y_test, predictions)
accuracy = acc.result().numpy()

pre = keras.metrics.Precision()
pre.update_state(y_test, predictions)
precision = pre.result().numpy()

rec = keras.metrics.Recall()
rec.update_state(y_test, predictions)
recall = rec.result().numpy()

print("Accuracy :", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)