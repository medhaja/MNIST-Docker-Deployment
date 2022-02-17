from flask import Flask, request, jsonify
import numpy as np
import json
from tensorflow import keras
from keras.models import model_from_json
app = Flask(__name__)
classes = {0: "Zero",
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "eight",
            9: "Nine"}

# load json and create model
json_file = open('Models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Models/model.h5")
print("Loaded model from disk")


@app.route('/predict', methods=['POST'])
def predict():
    parameters = request.get_json(force=True)
    im = np.array(parameters['image'])
    im = im.astype("float32") / 255
    im = np.expand_dims(im, -1)[None]
    out = classes[np.argmax(loaded_model.predict(im))]
    return out
if __name__ == '__main__':
    app.run(host='0.0.0.0')