import os, io, sys
import re
import numpy as np, csv

def inference_output(dirname, Actual_sentiment, predictor, writer):

    # converting a directory name to system directory
    listdir = os.listdir(dirname)

    # picking a random file name in directory
    for filename in listdir:
        review_filename = dirname + '/' + filename
        # Reading the content of file and appending to a list
        predOutput=[]
        with open(review_filename, 'rb') as txt_file:
            inputText = txt_file.read()
            predOutput.append(inputText)

        print filename
        prediction = predictor.predict(predOutput)


        if prediction[0] == 1:
            writer.writerow({'FileName': filename, 'PredictedSentiment': 'POSITIVE', 'ActualSentiment': Actual_sentiment})
        else:
            writer.writerow({'FileName': filename, 'PredictedSentiment': 'NEGATIVE', 'ActualSentiment': Actual_sentiment})