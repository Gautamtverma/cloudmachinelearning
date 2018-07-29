import argparse, os, json, csv

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from print_function import print_result

#calling the API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="apikey.json"

def analyze(posdirname, negdirname):
    # calling the API
    client = language.LanguageServiceClient()

    #Creating a CSV file including field names for out put storage
    with open('GCP_sentiment_Multiple_Folder.csv', 'w') as csvfile:
        fieldnames = ['FileName', 'SentimentScore', 'PredictedSentiment', 'ActualSentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        #Passing directories, actual sentiment, API and CSV writer to Print_Result function
        print_result(posdirname, 'POSITIVE', client, writer)
        print_result(negdirname, 'NEGATIVE', client, writer)

    print 'End of Detect Sentiment'


if __name__ == '__main__':
    #Accessing the Directory names from command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for Directory names and help information about Directory
    parser.add_argument('Pos_Directory_Name',help='The positive Folder of the review you\'d like to analyze.')
    parser.add_argument('Neg_Directory_Name', help='The negative Folder you need to of the review you\'d like to analyze.')

    args = parser.parse_args()
    #Calling the analyze function with parameters.
    analyze(args.Pos_Directory_Name, args.Neg_Directory_Name)
