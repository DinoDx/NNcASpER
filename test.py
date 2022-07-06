# For the use of the GPU with tensorflow
#import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

from tensorflow import keras
from dataPreprocessing import dataPreprocessing
import mlflow

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow_experiment_id = 0

# Load model and weights
with open("model/model.json", "r") as json_file:
    model_json = json_file.read()

model = keras.models.model_from_json(model_json)
model.load_weights("model/model.h5")

# Prepare test set 30%
x_test, y_test = dataPreprocessing(58000, None)

#with mlflow.start_run(experiment_id=mlflow_experiment_id):

predictions = (model.predict(x_test)).round()

acc = keras.metrics.CategoricalAccuracy()
acc.update_state(y_test, predictions)
accuracy = acc.result().numpy()
pre = keras.metrics.Precision()
pre.update_state(y_test, predictions)
precision = pre.result().numpy()

rec = keras.metrics.Recall()
rec.update_state(y_test, predictions)
recall = rec.result().numpy()

fmeasure = 2*((precision*recall)/(precision+recall))

print("Accuracy : ", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)
print("F-Measure : ", fmeasure) 


    #mlflow.log_param("n solutions", 20)
    #mlflow.log_param("n generations", 100)
    #mlflow.log_param("parent selection type", "sus")
    #mlflow.log_param("n elites", 2)
    #mlflow.log_param("crossover type", "uniform")
    #mlflow.log_param("crossover prob", 0.8)
    #mlflow.log_param("mutation type", "random")
    #mlflow.log_param("mutation prob", 0.1)

    #mlflow.log_metric("accuracy", accuracy)
    #mlflow.log_metric("precision", precision)
    #mlflow.log_metric("recall", recall)
    #mlflow.log_metric("f-measure", fmeasure)
