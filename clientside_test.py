import requests
from PIL import Image
import json
import os
import numpy as np
import io
import base64
import time

def clientside(test_size ="TINY",test_location = "local",test_batchsize =2, AWS_IP ="35.172.150.16"):
    """takes folder size, size of batch, test location and if AWS is given its IP address and sends post request for the image
    in groups of batchsize 
    """
    #folder_adress= "data/object-detection-TINY-unzip"

    # To implement: Test local vs AWS
    # To Implement: Test sending all at once (one lage JSON) vs one file at a time vs batches (50 - 100)


    ####################################################
    #####################################################
    # average upload time only makes sense for individual sends makes sense


    folder_adress= "data/object-detection-"+ test_size+"-unzip"

    if test_location=="AWS":
        # aws test with this, needs to be addapted with every lab restart (ip address changes)
        # 35.172.150.16
        url = 'http://'+ AWS_IP +'/api/detect'
    else:
        #local Test with this
        url = 'http://localhost:5000/api/detect'
    #35.172.150.16


    #needed for json format
    headers = {'Content-Type': 'application/json'}


    upload_times=[]
    responses = []

    i = 0
    start_time_upload = time.time()
    imgs=[]
    image_count=0
    batch_count=0
    # gp through all files/images in target folder 
    for filename in os.listdir(folder_adress):
        image_count+=1

        # for upload time calculation
        
        #open individual file
        with open(folder_adress+ "/" +filename, "rb") as imageFile:
            #read file
            image_data=imageFile.read()
            #convert to 64Bytes encoding
            image_data_BytesIO= io.BytesIO(image_data)
            pil_image = Image.open(image_data_BytesIO)
            # .encode(utf-8) makes sure that we send string, not bytes as json cant handle those
            imgs_byteversion=base64.b64encode(image_data).decode('utf-8')
            #add data to list that will be converted to json and send to via post request
            #  (structer kept the same for one image send as for multiple for consitancy and easy of use)
            imgs.append(imgs_byteversion)

        #need or for case that all files have been read
        if i == test_batchsize-1:
            
            batch_count+=1
            data = {'images': imgs}
            #send post request, get back json with box location and inference time

            #print("Befor Request")
            response = requests.post(url, headers=headers, data=json.dumps(data))
            #print("After Request")
            #print for test purposes
            #print(response.json())
            
            # save results (contains inverence time (and average inf time which can be ignored if only one image was send))
            responses.append(response)
            # measure endtime of batch
            end_time_upload = time.time()
            # calculate duration and save
            upload_times.append(end_time_upload-start_time_upload)
            start_time_upload = time.time()
            
            # reset batch
            imgs=[]
            i = 0

        else:
            i+=1


    # in case loop did not finish with a full batch, send the rest of unfull batch:

    if i != 0:
        batch_count+=1
        rest_batch_size=len(imgs)
        data = {'images': imgs}
        #send post request, get back json with box location and inference time
        response = requests.post(url, headers=headers, data=json.dumps(data))
        #print for test purposes
        #print(response.json())
        
        # save results (contains inverence time (and average inf time which can be ignored if only one image was send))
        responses.append(response)
        # measure endtime of batch
        end_time_upload = time.time()
        # calculate duration and save
        upload_times.append(end_time_upload-start_time_upload)

    else:
        rest_batch_size=0

    #print("Type of string is {}".format(type(imgs[0].decode('utf-8'))))
    #print(imgs[0])
    #print("Type of string is {}".format(type(imgs[0].decode('utf-8'))))
    #print(imgs[0].decode('utf-8'))
    avg_upload_time=sum(upload_times)/len(upload_times)


    # not sure if this is gonna work out... depends if json simplefies short lists with only one element:
    # maybe json function needed

    inf_times=[]
    boundig_boxes= []
    for tmpresponse in responses:
        #print(tmpresponse.json())
        #print("json dumps")
        #print(json.dumps(tmpresponse.json()))
        tmpjson= tmpresponse.json()

        for j in range(len(tmpjson['inf_time'])):
            inf_times.append(tmpjson['inf_time'][j])
            boundig_boxes.append(tmpjson['bounding_boxes'][j])


    avg_inf_time=sum(inf_times)/len(inf_times)

    final_dict_keydata={'avg_inf_times':avg_inf_time,
                'avg_upload_times': avg_upload_time,
                'image_count':image_count,
                'batch_size': test_batchsize,
                'calc_location': test_location}

    final_dict_results= {'upload_times': upload_times,
                        'inf_times': inf_times,
                        'bounding_boxes':boundig_boxes}


    resultfile_path= "results/test_calcAt_" + test_location+"_BatchS_"+ str(test_batchsize) + "_Folder_"+ test_size

    resultfile_keydata_path= resultfile_path + "keydata"

    with open(resultfile_path, 'w') as fout:
        json.dump(final_dict_results, fout)

    with open(resultfile_keydata_path, 'w') as fout:
        json.dump(final_dict_keydata, fout)



if __name__ == '__main__':
    
    test_sizes= [
    #            "TINY"
               "SMALL"
    #            "MEDIUM"
    #            "BIG"
                ]

    # if AWS remeber to update ip adress with each lab restart!!

    #test_locations =["local","AWS"]
    test_locations=[]
    #test_locations.append("local")
    test_locations.append("AWS")


    AWS_IP="34.236.170.117"

    for test_size in test_sizes:
        for test_location in test_locations:
            
            if test_size=="SMALL":
                test_batchsizes = [10,50,100,500]
            elif test_size == "MEDIUM":
                test_batchsizes = [50,100,500,1000]
            elif test_size == "BIG":
                test_batchsizes = [200,500,1000]
            else:
                test_batchsizes = [100]

            for test_batchsize in test_batchsizes:
                clientside(test_size=test_size,test_location=test_location,test_batchsize=test_batchsize,AWS_IP=AWS_IP)



###############################################
###############################################
"""
if test_type == "all":

    #35.172.150.16

    upload_times=[]
    imgs=[]
    start_time_upload = time.time()
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


    # 35.172.150.16


    #url = 'http://35.172.150.16:5000/api/detect'
    url = 'http://localhost:5000/api/detect'

    headers = {'Content-Type': 'application/json'}
    data = {'images': imgs}
    # get one single response for all different pictures
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
    end_time_upload = time.time()
    upload_times.append(end_time_upload-start_time_upload)





#########################################################################################
########################################################################################
# individual
else:
    upload_times=[]
    responses = []

    # gp through all files/images in target folder 
    for filename in os.listdir(folder_adress):
        # for upload time calculation
        start_time_upload = time.time()
        #conver single image into file and save results thus reset imgs in every loop
        imgs=[]
        #open individual file
        with open(folder_adress+ "/" +filename, "rb") as imageFile:
            #read file
            image_data=imageFile.read()
            #convert to 64Bytes encoding
            image_data_BytesIO= io.BytesIO(image_data)
            pil_image = Image.open(image_data_BytesIO)
            # .encode(utf-8) makes sure that we send string, not bytes as json cant handle those
            imgs_byteversion=base64.b64encode(image_data).decode('utf-8')
            #add data to list that will be converted to json and send to via post request
            #  (structer kept the same for one image send as for multiple for consitancy and easy of use)
            imgs.append(imgs_byteversion)
        data = {'images': imgs}
        #send post request, get back json with box location and inference time
        response = requests.post(url, headers=headers, data=json.dumps(data))
        #print for test purposes
        #print(response.json())
        
        # save results (contains inverence time (and average inf time which can be ignored if only one image was send))
        responses.append(response)
        # measure endtime
        end_time_upload = time.time()
        # calculate duration and save
        upload_times.append(end_time_upload-start_time_upload)
    #print("Type of string is {}".format(type(imgs[0].decode('utf-8'))))
    #print(imgs[0])
    #print("Type of string is {}".format(type(imgs[0].decode('utf-8'))))
    #print(imgs[0].decode('utf-8'))
    avg_upload_time=sum(upload_times)/len(upload_times)


"""
##############################################################
##############################################################
# approch giving batch sizes:



    # To implement: Test local vs AWS
    # To Implement: Test sending all at once (one lage JSON) vs one file at a time

