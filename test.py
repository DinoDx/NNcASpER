from tensorflow import keras
from dataPreprocessing import dataPreprocessing

#Load model and weights
with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = keras.models.model_from_json(model_json)
model.load_weights("model.h5")

# Prepare test set 30%
x_test, y_test = dataPreprocessing(5800, None)

predictions = model.predict(x_test)
bce = keras.losses.BinaryCrossentropy()
print("Binary Crossentropy : ", bce(y_test, predictions).numpy())

acc = keras.metrics.BinaryAccuracy(threshold = 0.5)
acc.update_state(y_test, predictions)
accuracy = acc.result().numpy()
print("Accuracy :", accuracy)



