import io, os, argparse, json, csv
from google.cloud import vision, pubsub
from google.cloud.vision import types
from image_analysis import image_analysis

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="apikey.json"

def vision_analysis(dirname):
    #calling the Google vision API
    client = vision.ImageAnnotatorClient()
    #reading a file
    # Creating a CSV file including field names for out put
    with open('Google_Image_Analysis.csv', 'w') as csvfile:
        fieldnames = ['FileName','Prediction']
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
    vision_analysis(args.Directory_Name)
