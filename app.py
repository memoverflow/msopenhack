# -*- coding:utf-8 -*-
__author__ = 'ABB'
__depart__ = u'Ability'

import os
import uuid
import json
import cv2

from predict import *
from flask import Flask, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

predict_folder = "/home/contoso/notebooks/XRCode/data/predict/"

class predict(Resource):
     def post(self):
        file = request.files['file']
        print(file.filename)
        extension = os.path.splitext(file.filename)[1]
        f_name = str(uuid.uuid4()) + extension
        print(f_name)
        filename = os.path.join(os.getcwd(), f_name)
        file.save(filename)
        real_path = (predict_folder+f_name).replace('.jpg','.png').replace('.jpeg','.png')
        print(predict)
        convert_img(filename,real_path)
        result = predict_img(real_path)
        return json.dumps({'predict':result})

api.add_resource(predict, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
