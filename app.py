# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from PIL import ImageDraw
import os
#import detect
#import tflite_runtime.interpreter as tflite
import platform
import datetime
#import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify, make_response
import random
import re
import base64
import tensorflow as tf
#needed to download the model
import tensorflow_hub as hub

import tempfile


#initializing the flask app
app = Flask(__name__)
"""
def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)
"""

def load_img(path):
  #  get image from temp save
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  # turn all messegaes back on
  return img


def detection_loop(filenames_images):
  #module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
  # only one that does not kill my computer
  module_handle= "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
  #module_handle= "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
  detector = hub.load(module_handle).signatures['default']

  """ Test later for performance improvment via local saving of the model
  saved_model_path= "./ts-detectionmodel"
  if os.path.exists(saved_model_path):
    print('load model from local file')
    detector = hub.load(saved_model_path).signatures['default']
  else:
    print('load model from hub online')
    detector = hub.load(module_handle).signatures['default']
    tf.saved_model.save(detector, saved_model_path)
  """
  #results= []

  #display= False
  bounding_boxes=[]
  inf_times=[]
  #upload_times=[]

  for filename in filenames_images:
    #results.append(run_detector(detector=detector, path=filename))

    #start_time_upload = time.time()
    img = load_img(filename)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time_inf = time.time()
    #start actual detection
    result = detector(converted_img)
    end_time_inf = time.time()

    #end_time_upload = time.time()
    result= {key:value.numpy().tolist() for key,value in result.items()}

    """
    #not needed as we only want bounding boxes and inference time

    detection_class_names_list=[]
    detection_class_entities_list=[]
    # decode the byte strings to make them json serilizable
    for i in range(len(result['detection_class_names'])):
      detection_class_names_list.append(result['detection_class_names'][i].decode('utf-8'))
      detection_class_entities_list.append(result['detection_class_entities'][i].decode('utf-8'))
    result['detection_class_names']=detection_class_names_list
    result['detection_class_entities']=detection_class_entities_list
    """

    bounding_boxes.append(result['detection_boxes'])
    inf_times.append(end_time_inf-start_time_inf)

    #upload_times.append(end_time_upload-start_time_upload)

    """
    if display:
      print("Found %d objects." % len(result["detection_scores"]))
      print("Inference time: ", end_time_inf-start_time_inf)
      image_with_boxes = draw_boxes(img.numpy(), result["detection_boxes"],result["detection_class_entities"], result["detection_scores"])
      display_image(image_with_boxes)
    """

  #avg_inf_time= sum(inf_times)/len(inf_times)
  #avg_upload_time=sum(upload_times)/len(upload_times)
  
  #convert to list to make serilizable

  data = {
      "status": 200,
      "bounding_boxes": bounding_boxes,
      "inf_time": inf_times,
      #"avg_inf_time": str(avg_inf_time),
      #"upload_time": upload_times,
      #"avg_upload_time": str(avg_upload_time), 
  }  


  result_json=jsonify(data)
  result_make_response= make_response(result_json, 200)
  return result_make_response

#routs for testing if server is running and if it can return json
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route("/hello", methods=["GET"])
def say_hello():
    return jsonify({"msg": "Hello from Flask"})

#routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  data= request.get_json(force = True)
  #get the array of images from the json body
  imgs = data['images']

  filename_image =[]
  #new_width=256
  #new_height=256
  for img in imgs:
      _, filename = tempfile.mkstemp(suffix=".jpg")
      #decode image 
      pil_image=Image.open(io.BytesIO(base64.b64decode(img)))
      #pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
      pil_image_rgb = pil_image.convert("RGB")
      #save intermidiatly to make it easy to open later in detection loop
      pil_image_rgb.save(filename, format="JPEG", quality=90)
      filename_image.append(filename)
  result_detectionloop=detection_loop(filename_image)

  
  return result_detectionloop
  
# status_code = Response(status = 200)
#  return status_code
# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
    #test loading model here for performance improvment?
