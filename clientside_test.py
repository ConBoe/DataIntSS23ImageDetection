from flask import Request
import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
# for drawing boxes
from PIL import ImageColor
from PIL import ImageFont

import json
# for testing
#from flask import jsonify

# to save file temporatily which seems to be suggested by the passing of image_filename in template 
# i.e. return detection_loop(filename_image)
import tempfile


import os
# ? is this a actual package or their way to say we
#  should built a package in the docker 
#import detect

# only available for certain python version and not sure whate use
#import tflite_runtime.interpreter as tflite
import platform
import datetime
# included in opencv üackage
import cv2
import time
import numpy as np
import io
from io import BytesIO
#from flask import Flask, request, Response, jsonify
import random
import re
import base64
import matplotlib.pyplot as plt
import tensorflow as tf
#needed to download the model
import tensorflow_hub as hub


url = 'http://128.130.246.82:5000/api/detect'  # Replace with the URL of your Flask app


imgs=[]
for filename in os.listdir("data/object-detection-TINY-unzip"):
    with open("data/object-detection-TINY-unzip/"+filename, "rb") as imageFile:
        image_data=imageFile.read()
        image_data_BytesIO= io.BytesIO(image_data)
        pil_image = Image.open(image_data_BytesIO)
        imgs_byteversion=base64.b64encode(image_data)
        imgs.append(imgs_byteversion)

data= {}
data['images']=imgs

response = requests.post(url,)

if response.status_code == 200:
    print('Request succeeded')
else:
    print('Request failed')