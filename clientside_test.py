import requests
from PIL import Image
import json
import os
import numpy as np
import io
import base64
folder_adress= "data/object-detection-TINY-unzip"

imgs=[]
for filename in os.listdir(folder_adress):
    with open(folder_adress+ "/" +filename, "rb") as imageFile:
        image_data=imageFile.read()
        image_data_BytesIO= io.BytesIO(image_data)
        pil_image = Image.open(image_data_BytesIO)
        # .encode(utf-8) makes sure that we send string, not bytes as json cant handle those
        imgs_byteversion=base64.b64encode(image_data).decode('utf-8')
        imgs.append(imgs_byteversion)

#print("Type of string is {}".format(type(imgs[0].decode('utf-8'))))
#print(imgs[0])
#print("Type of string is {}".format(type(imgs[0].decode('utf-8'))))
#print(imgs[0].decode('utf-8'))

url = 'http://localhost:5000/api/detect'
headers = {'Content-Type': 'application/json'}
data = {'images': imgs}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())