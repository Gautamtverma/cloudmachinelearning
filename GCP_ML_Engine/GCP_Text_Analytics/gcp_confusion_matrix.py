from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pandas_ml import ConfusionMatrix
import pandas as pd, numpy as np
import argparse, os, json

def calculate_accuracy(csv_filename):

    # Loading csv information into a data frame
    data = pd.read_csv(csv_filename)
    # assigning actual sentiment data to y_test
    y_test = data['ActualSentiment']
    # assigning predicted sentiment data to y_pred
    y_pred = data['PredictedSentiment']

    score = accuracy_score(y_test, y_pred)
    # calling accuracy_score method to get the accuracy_score
    print 'Accuracy Score : ', score

    # calling confusion_matrix method from pandas_ml to show the output
    confusion_matrix = ConfusionMatrix(y_test, y_pred)
    output = confusion_matrix.to_dataframe()

    writer = pd.ExcelWriter("gcp_vm_text_confusion_matrix_output.xlsx")
    output.to_excel(writer, startrow=4, startcol=0)
    Acuracy_Score = 'Accuracy Score : ' + str(score)
    worksheet = writer.sheets['Sheet1']
    worksheet.write(1, 0, Acuracy_Score)

    writer.save()

    print("Confusion matrix:\n%s" % confusion_matrix)


if __name__ == '__main__':
    # Accessing the file name as a parameter from command line
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for file name and help information about file
    parser.add_argument(
        'csv_filename',
        help='The filename of the movie review you\'d like to analyze.')
    args = parser.parse_args()
    # Calling the calculate_accuracy function with parameters.
    calculate_accuracy(args.csv_filename)
