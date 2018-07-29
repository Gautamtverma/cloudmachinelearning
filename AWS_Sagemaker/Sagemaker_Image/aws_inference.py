
# coding: utf-8

# In[1]:


import argparse, os, io
import re, csv
import sagemaker, boto3
from sagemaker.mxnet.model import MXNetModel
import logging
logging.getLogger().setLevel(logging.WARNING)

from scipy import io as sio
import numpy as np

from sagemaker.tensorflow.model import TensorFlowModel

import cv2


# In[13]:


def load_model():
    sagemaker_model = TensorFlowModel(
        role='arn:aws:iam::821488635735:role/service-role/AmazonSageMaker-ExecutionRole-20171216T132860',
        model_data='s3://sagemaker-us-east-2-821488635735/object_detection/model.tar.gz',
        entry_point='object_detect.py')
    
    predictor = sagemaker_model.deploy(initial_instance_count=1,
                                       instance_type='ml.m4.xlarge')
    print ("Model Loaded, Now start predicting")
    return predictor


# In[15]:


def load_data(filename):
    # load the data 
    data = sio.loadmat(filename)
    images = data['images']
    labels = data['labels']
    names = data['Names']
    
    return images, labels, names


# In[9]:


def create_names_list(names):
    nms = []
    for name in names:
        nms.append(name[0][0])
    return nms


# In[25]:


def predict_data(im, predictor):
    pred_out = predictor.predict(data=np.float32(im)/255.)
    
    probs = np.array(pred_out['outputs']['classes']['floatVal'])
    
    
    return probs


# In[20]:


def pre_process_data(img, IMG_SIZE):
    im = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    im = np.float32(im)/255.
    
    return im


# In[26]:


def predict_test_data(filename):
    predictor = load_model()
    ext = filename.split('.')[-1]
    if ext == 'mat':
    
        images, labels, names = load_data(filename)

        pos = 0
        neg = 0
        with open('Aws_Complete_image_analysis_new.csv', 'w') as csvfile:
            fieldnames = ['FileName', 'Predicted_Lable', 'Original_Label', 'Predicted_Correct_Label',
                              'Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(images.shape[0]):
                im = pre_process_data(images[idx, :, :, :], 80)

                lab = labels[idx, :]

                probs = predict_data(im, predictor)
                nms = names[probs >= 0.5]
                nms = create_names_list(nms)
                actual = names[lab == 1.]
                actual = create_names_list(actual)
                cur_pos = 0
                totpred = []
                for nm in nms:
                    for act in actual:
                        if nm == act:
                            totpred.append(act)
                            pos+=1
                            cur_pos+=1
                accuracy = cur_pos * 1.0 /len(actual)
                writer.writerow({'FileName': str(idx), 'Predicted_Lable': nms, 'Original_Label': actual,
                                             'Predicted_Correct_Label': totpred, 'Accuracy': accuracy})
                neg+=(len(actual) - cur_pos)
    elif(ext != 'mat'):
        img = cv2.imread(filename)
        
        im = pre_process_data(img)
        nms = predict_data(im, predictor)
        print ('prediction output = ', nms)


# In[ ]:


if __name__ == '__main__':
    # Accessing the file names from command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for Directory names and help information about Directory
    parser.add_argument('FileName',help='image or mat filename.')

    args = parser.parse_args()
    # Calling the sentiment_detect function with parameters.
    predict_test_data(args.FileName)

