import numpy as np, argparse, os, io
import tensorflow as tf
import re, csv
from inference_output import inference_output


def sentiment_analysis_inference(posdirname, negdirname):

    with open('GCP_New_train-5000.csv', 'w') as csvfile:
        fieldnames = ['FileName', 'SentimentScore', 'PredictedSentiment', 'ActualSentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Passing directories, actual sentiment, API and CSV writer to compute_sentiment_for_folder function
        inference_output(posdirname, 'POSITIVE', writer)
        inference_output(negdirname, 'NEGATIVE', writer)

if __name__ == '__main__':
    # Accessing the file names from command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for Directory names and help information about Directory
    parser.add_argument('Pos_Directory_Name',help='The positive Folder of the review you\'d like to analyze.')
    parser.add_argument('Neg_Directory_Name', help='The negative Folder you need to of the review you\'d like to analyze.')

    args = parser.parse_args()
    # Calling the sentiment_detect function with parameters.
    sentiment_analysis_inference(args.Pos_Directory_Name, args.Neg_Directory_Name)
