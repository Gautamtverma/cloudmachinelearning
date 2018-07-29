	#Importing libraries
import tensorflow as tf    # ML library for graphs
import cv2                 # image processing
import numpy as np         # mathematical operations
import os                  # working with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from scipy import io as sio
import csv
import argparse

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="apikey.json"

IMG_SIZE = 80# our images are 80x80x3
NumClasses = 10

steps = 50
epochs = 100
step_size = 20
total_batch = int(steps/step_size)
LR = 0.00001
#model_path = 'gs://image-analytics/New-model-100/'
#csv_file_name = 'GCP_100Image_analysis.csv'

model_path = 'gs://image-analytics/New-model-400/'
csv_file_name = 'GCP_400Image_analysis.csv'

def load_data(path):
    TrainData = sio.loadmat(path)
    print("loded the Images")
    train_data = TrainData['images']
    print(train_data.shape)
    names = TrainData['Names']
    # print(names)
    train_labels = TrainData['labels']
    
    #Splitting train and CV data
    train = train_data
        
    X = np.zeros((len(train), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    Y = train_labels[0:len(train), :]
    idx = 0
    for i in train:
        img = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
        X[idx, :, :, :] = img
        idx+=1       
   
    
    X = X.astype(np.float32)/255.
    Y = Y.astype(np.float32)

    return X, Y, names


def Graph_init():
    #FUNCTIONS TO CODE LAYERS
    def init_weights(shape):
        init_random_dist = tf.zeros(shape)
        return tf.Variable(init_random_dist)

    def init_bias(shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2by2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def convolutional_layer(input_x, shape):
        W = init_weights(shape)
        b = init_bias([shape[3]])
        return tf.nn.relu(conv2d(input_x, W) + b)

    def normal_full_layer(input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size])
        b = init_bias([size])
        return tf.matmul(input_layer, W) + b



    #GRAPH
    with tf.Session() as sess:
        g = tf.Graph()

    #Defining Placeholdera
    x = tf.placeholder(dtype=tf.float32,shape=[None,80,80,3], name='train')
    y_true = tf.placeholder(dtype=tf.float32,shape=[None,NumClasses], name='label')

    #Defining GRAPH
    with tf.name_scope('Model'):
        convo_1 = convolutional_layer(x,shape=[6,6,3,32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_3 = convolutional_layer(convo_2_pooling,shape=[6,6,64,64])
        convo_3_pooling = max_pool_2by2(convo_3)
        convo_4 = convolutional_layer(convo_3_pooling,shape=[6,6,64,128])
        convo_4_pooling = max_pool_2by2(convo_4)
        convo_2_flat = tf.reshape(convo_4_pooling,[-1,5*5*128])

        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,4096))
        hold_prob1 = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob1)

        full_layer_two = tf.nn.relu(normal_full_layer(full_one_dropout,2048))
        hold_prob2 = tf.placeholder(tf.float32)
        full_two_dropout = tf.nn.dropout(full_layer_two,keep_prob=hold_prob2)

        full_layer_three = tf.nn.relu(normal_full_layer(full_two_dropout,1024))
        hold_prob3 = tf.placeholder(tf.float32)
        full_three_dropout = tf.nn.dropout(full_layer_three,keep_prob=hold_prob3)

        full_layer_four = tf.nn.relu(normal_full_layer(full_three_dropout,512))
        hold_prob4 = tf.placeholder(tf.float32)
        full_four_dropout = tf.nn.dropout(full_layer_four,keep_prob=hold_prob4)

        y_pred = normal_full_layer(full_four_dropout,NumClasses)
    return x, y_true,hold_prob1,hold_prob2, hold_prob3, hold_prob4, y_pred


def get_names_pred(pred, names):
    nms = []
    maxval = np.amax(pred)
    pred = pred / maxval
    for idx in range(len(pred)):
        p = pred[idx]
        
        if p > 0.5:
            
            nms.append(names[idx][0][0])
    return nms

def get_names_actual(act, names):
    nms = []
    for idx in range(len(act)):
       
        if act[idx] == 1.0:
            nms.append(names[idx][0][0])
    return nms



def test(datapath):
    # initialize the graph
    x, y_true, hold_prob1,hold_prob2, hold_prob3, hold_prob4, y_pred = Graph_init()
    
    # get the data
    X, Y, names = load_data(datapath)
    
        #Writing Loss and Accuracy
    with tf.name_scope('Loss'):
        cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_true, y_pred))

    with tf.name_scope('SGD'):
        train = tf.train.AdamOptimizer(learning_rate=LR).minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
    #     matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))    

        y_pred_out = tf.greater(y_pred, 0.5)
        y_pred_out = tf.cast(y_pred_out, tf.float32)
        matches = tf.equal(y_pred_out, y_true)
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    init = tf.global_variables_initializer()

    tf.summary.scalar("loss", cross_entropy)
    tf.summary.scalar("accuracy", acc)

    merged_summary_op = tf.summary.merge_all()
    print(merged_summary_op.graph)
    acc_list = []
    cross_entropy_list = []
    acc_train = []

    saver = tf.train.Saver()
   
    # RESTORING and MAKING PREDICTIONS FOR FIRST 64 IMAGES
    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(model_path))
        print("Model restored.") 
        print('Initialized')
        with open(csv_file_name, 'w') as csvfile:
            fieldnames = ['FileName', 'Original_Label', 'Predicted_Correct_Label',
                              'Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for id in range(X.shape[0]):
                k = session.run([y_pred], feed_dict={x:X[id, :, :, :].reshape((1, 80, 80, 3)) , hold_prob1:1,hold_prob2:1,hold_prob3:1,hold_prob4:1})

                pred = k[0][0]
                pred_names = get_names_pred(pred, names)
                actual_names = get_names_actual(Y[id, :], names)
                
                cur_pos = 0
                totpred = []
                for nm in pred_names:
                    for act in actual_names:
                        if nm == act:
                            totpred.append(act)
                            cur_pos+=1
                accuracy = cur_pos * 100.0 /len(actual_names)
                tot_pred = np.unique(totpred)
                print(str(id))
                writer.writerow({'FileName': str(id), 'Original_Label': actual_names,
                                             'Predicted_Correct_Label': tot_pred, 'Accuracy': accuracy})


        
        
if __name__ == '__main__':
    # Accessing the file names from command line
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Adding arguments for Directory names and help information about Directory
    parser.add_argument('FileName',help='image or mat filename.')

    args = parser.parse_args()

    test(args.FileName)
