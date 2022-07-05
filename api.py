import re
import flask
import numpy as np
from sklearn import preprocessing
from tensorflow import keras

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])

def home():
    args = flask.request.args
    metrics_arg = args.get('metrics')
    metrics = np.array([float(num) for num in re.findall(r'-?\d+\.?\d*', metrics_arg)])
    if(metrics.size != 19):
        return "Error: 19 float metrics required, " + str(metrics.size) + " provided!"
    prediction = classify(metrics = metrics)

    toreturn = ""
    for pred in prediction:
        toreturn += str(pred) + ""

    return toreturn

def classify(metrics):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    metrics = scaler.fit_transform(metrics.reshape(19, 1))
    metrics = metrics.reshape(1,19)

    #Load model and weights
    with open("model/model.json", "r") as json_file:
        model_json = json_file.read()

    model = keras.models.model_from_json(model_json)
    model.load_weights("model/model.h5")

    prediction = model.predict(metrics).round()
    
    return prediction

app.run()