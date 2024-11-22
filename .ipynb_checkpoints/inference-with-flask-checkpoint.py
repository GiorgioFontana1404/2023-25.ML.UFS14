import logging
import json
import glob
import sys
from os import environ
from flask import Flask
import joblib
import sklearn
import numpy as np
from flask import request

logging.debug('Init a Flask app')
app = Flask(__name__)


def doit():
    model = joblib.load("output.joblib")
    predict_input = np.array([
        [Square_Footage,Num_Bedrooms,Num_Bathrooms,Year_Built,Lot_Size,Garage_Size,Neighborhood_Quality,3058,2,1,2017,1.498552073529678,0,2]
    ])
    predict_result = model.predict(predict_input)

    return json.dumps({"predict_result": predict_result.tolist()})

@app.route('/ping')
def ping():
    logging.debug('Hello from route /ping')

    return doit()