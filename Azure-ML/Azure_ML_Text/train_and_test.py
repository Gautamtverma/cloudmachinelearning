from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pandas_ml import ConfusionMatrix
import pandas as pd, numpy as np
import argparse, os, json
import csv

# Use the Azure Machine Learning data collector to log various metrics
from azureml.logging import get_azureml_logger
import os
import warnings
logger = get_azureml_logger()
# Log cell runs into run history
logger.log('Cell','Set up run')
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import tatk
from tatk.pipelines.text_classification.text_classifier import TextClassifier

def Sentiment_Training(posdir, negdir):
    # Training Dataset Location
    # training_file_path = 'train-1000.csv'
    training_file_path = 'train-24500.csv'
    poslistdir = os.listdir(posdir)
    neglistdir = os.listdir(negdir)
    pos = 'POSITIVE'
    neg = 'NEGATIVE'

    
    df_train = pd.read_csv(training_file_path,
                       sep = ',') 
    df_train=df_train[df_train.columns[0:2]]

    print("df_train.shape= {}".format(df_train.shape))
    # Define the scikit-learn estimator.
    log_reg_learner = LogisticRegression(penalty='l2', dual=False,
                                     tol=0.0001, C=1.0, fit_intercept=True,
                                     intercept_scaling=1, class_weight=None,
                                     random_state=None, solver='lbfgs', 
                                     max_iter=1000, multi_class='ovr',
                                     verbose=1, warm_start=False, n_jobs=3) 

    # Define the text classifier model.
    # It will fit the learner on the text column "text".
    text_classifier = TextClassifier(estimator=log_reg_learner, 
                                 text_cols = ["Text"], 
                                 label_cols = ["Label"], 
#                                numeric_cols = None,
#                                cat_cols = None, 
                                 extract_word_ngrams=True,
                                 extract_char_ngrams=True)
    text_classifier.fit(df_train)
    text_classifier.get_step_param_names_by_name("Text_word_ngrams")
    pos_file = r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Pos_Info.csv'
    with open(r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Pos_Info.csv', 'w') as csvfile:
        fieldnames = ['Filename', 'Label', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        id =0
        for name in poslistdir:
            filename = posdir + name
            data = []
            print(name)
            with open(filename, 'r') as txtf:
                data.append(txtf.readlines())

            writer.writerow({'Filename':name, 'Label': pos, 'Text': data[0][0]})
    print("loded positive entries into excel")
    neg_file = r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Neg_Info.csv'
    with open(r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Neg_Info.csv', 'w') as csvfile:
        fieldnames = ['Filename', 'Label', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        id =0
        for name in neglistdir:
            filename = negdir + name
            data = []
            print(name)
            with open(filename, 'r') as txtf:
                data.append(txtf.readlines())

            writer.writerow({'Filename':name, 'Label': neg, 'Text': data[0][0]})
    print("loded negative entries into excel")
    # Training Dataset Location
    test_file_path = r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Pos_Info.csv'
    df_test = pd.read_csv(r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Pos_Info.csv', sep = ',')
    pred_df = pd.Series.to_frame(df_test.iloc[:, 2])
    pred_df = text_classifier.predict(pred_df)

    output = r' C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Azure_sentiment_Analysis_24500R.csv'
    with open(r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Azure_sentiment_Analysis_24500R.csv', 'a') as csvfile:
        fieldnames = ['Filename', 'Score', 'Prediction', 'Actual_Statement']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(pred_df)):
            if pred_df['prediction'][idx]== 1:
                writer.writerow({'Filename':df_test['Filename'][idx], 'Score': pred_df['probabilities'][idx], 'Prediction':'POSITIVE', 'Actual_Statement':df_test['Label'][idx]})
            else:
                writer.writerow({'Filename':df_test['Filename'][idx], 'Score': pred_df['probabilities'][idx], 'Prediction':'NEGATIVE', 'Actual_Statement':df_test['Label'][idx]})
        print("writtern positive entries into excel")  
        
        test_file_path = 'Test_Neg_Info.csv'
        ndf_test = pd.read_csv(r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Test_Neg_Info.csv', sep = ',')
        npred_df = pd.Series.to_frame(ndf_test.iloc[:, 2])
        npred_df = text_classifier.predict(npred_df)
        
        for idx in range(len(npred_df)):
            if npred_df['prediction'][idx]== 1:
                writer.writerow({'Filename':ndf_test['Filename'][idx], 'Score': npred_df['probabilities'][idx], 'Prediction':'POSITIVE', 'Actual_Statement':ndf_test['Label'][idx]})
            else:
                writer.writerow({'Filename':ndf_test['Filename'][idx], 'Score': npred_df['probabilities'][idx], 'Prediction':'NEGATIVE', 'Actual_Statement':ndf_test['Label'][idx]})
    print("written negative entries into excel")
    # Loading csv informtion into a data frame
    data = pd.read_csv(r'C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Azure_sentiment_Analysis_24500R.csv')
    y_test = data['Actual_Statement']
    # assigning predicted sentiment data to y_pred
    y_pred = data['Prediction']

    score = accuracy_score(y_test, y_pred)
    # calling accuracy_score method to get the accuracy_score
    print('Accuracy Score : ', score)

    # calling confusion_matrix method from pandas_ml to show the output
    confusion_matrix = ConfusionMatrix(y_test, y_pred)
    output = confusion_matrix.to_dataframe()

    writer = pd.ExcelWriter(r"C:\Users\admin-dsvm\Downloads\tap-1.0.0b3-release\tap-1.0.0b3-release\notebooks\Azure_5000_itr_output.xlsx")
    output.to_excel(writer, startrow=4, startcol=0)
    Acuracy_Score = 'Accuracy Score : ' + str(score)
    worksheet = writer.sheets['Sheet1']
    worksheet.write(1, 0, Acuracy_Score)

    writer.save()

    print("Confusion matrix:\n%s" % confusion_matrix)
    
       
if __name__ == '__main__':
    #Accessing the parameter from command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for Directory name and help information about Directory
    parser.add_argument('Pos_Directory_Name',help='The positive Folder of the review you\'d like to analyze.')
    parser.add_argument('Neg_Directory_Name',help='The positive Folder of the review you\'d like to analyze.')

    args = parser.parse_args()
    #Calling the sentiment_detect function with parameters.
    Sentiment_Training(args.Pos_Directory_Name, args.Neg_Directory_Name)