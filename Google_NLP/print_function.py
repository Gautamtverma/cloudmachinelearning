import os, csv, json
from google.cloud.language import enums
from google.cloud.language import types

def print_result(dirname, Actual_sentiment, client, writer):
    #converting a string to a directory
    listdir = os.listdir(dirname)

    #picking a random file name in directory
    for filename in listdir:
        review_filename = dirname + filename

        #Reading the content of file
        with open(review_filename, 'r') as review_file:
            content = review_file.read()

            print 'filename', filename

            #Converting entire document to a plain text
            document = types.Document(
                content=content,
                type=enums.Document.Type.PLAIN_TEXT)
            #API will remove the general words and analyze the document
            annotations = client.analyze_sentiment(document=document)
            #Get the sentiment score of entire document
            score = annotations.document_sentiment.score

            # writing the information to csv file based on score
            if score < 0:
                writer.writerow({'FileName': filename, 'SentimentScore': score,
                                 'PredictedSentiment': 'NEGATIVE', 'ActualSentiment': Actual_sentiment})
            elif score == 0:
                writer.writerow({'FileName': filename, 'SentimentScore': score,
                                 'PredictedSentiment': 'NEUTRAL', 'ActualSentiment': Actual_sentiment})
            else:
                writer.writerow({'FileName': filename, 'SentimentScore': score,
                                 'PredictedSentiment': 'POSITIVE', 'ActualSentiment': Actual_sentiment})
