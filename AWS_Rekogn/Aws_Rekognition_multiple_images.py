import boto3
import csv, argparse, os
from image_analysis import image_analysis

def aws_rekognition(dirname):
    #calling the AWS rekognition API
    client = boto3.client('rekognition')

    # reading a file
    # Creating a CSV file including field names for out put
    with open('Aws_Image_Analysis.csv', 'w') as csvfile:
        fieldnames = ['FileName', 'Prediction', 'Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Passing directory name, API and CSV writer to image_analysis function
        image_analysis(dirname, client, writer)

    print 'End of Image Analysis'

if __name__ == '__main__':
    # Accessing the directory name from command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for directory name and help information about files
    parser.add_argument('Directory_Name',help='The images Folder which you\'d like to analyze.')

    args = parser.parse_args()
    # Calling the vision_analysis function with parameter.
    aws_rekognition(args.Directory_Name)