import flask
import numpy as np
from tensorflow import keras

from dataPreprocessing import dataPreprocessing

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])

def home():
    args = flask.request.args
    metrics_arg = args.get('metrics')
    metrics = np.array(metrics_arg)
    metrics = dataPreprocessing(metrics.astype(float))
    prediction = classify(metrics = metrics)
    
    return flask.jsonify(prediction)

def classify(metrics):
    #Load model and weights
    with open("model/model.json", "r") as json_file:
        model_json = json_file.read()

    model = keras.models.model_from_json(model_json)
    model.load_weights("model/model.h5")

    prediction = model.predict(metrics).round()
    
    return prediction

app.run()