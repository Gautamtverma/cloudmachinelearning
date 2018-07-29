import os, csv, json, io
from google.cloud import vision, pubsub
from google.cloud.vision import types

def image_analysis(dirname, client, writer):
    #converting a string to directory
    listdir = os.listdir(dirname)
    #picking a random file name in directory
    for filename in listdir:
        review_filename = dirname +'/'+ filename
        #Reading the content of file
        with io.open(review_filename, 'rb') as image_file:
            content = image_file.read()
        # converting a file into Image type
        image = types.Image(content=content)
        # Detecting the labels in image
        response = client.label_detection(image=image)
        # Getting the lables
        labels = response.label_annotations
        output = []
        # storing labels description information in output variable
        for label in labels:
            output.append(str(label.description))

        print 'filename', filename
        # writing the file name and predicted objects information to csv file
        writer.writerow({'FileName':filename, 'Prediction':output})
