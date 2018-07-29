import os, csv
from writing_to_csv import writing_data_tocsv

def compare_label():
    imgData = []
    gtData = []
    #Reading predicted label csv file and adding the data to imgData
    with open('Aws_Image_Analysis.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_file)
        for line in csv_reader:
            imgData.append(line)
            # print line
    # Reading Original label csv file and adding the data to gtData
    with open('GroundTruth.csv', 'r') as gt_csv:
        csv_reader1 = csv.reader(gt_csv)
        next(gt_csv)
        for lines in csv_reader1:
            gtData.append(lines)
            # print lines
    #creating a aws complete image analysis csv file along with the field names
    with open('Aws_Complete_image_analysis.csv', 'w') as csvfile:
        fieldnames = ['FileName', 'Predicted_Lable', 'Original_Label', 'Predicted_Correct_Label',
                      'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Sending Predicted labels, Ground truth labels and writer to writing_data_tocsv function
        writing_data_tocsv(imgData, gtData, writer)

    print 'Comparision has been completed'
compare_label()
