#Importing libraries
import tensorflow as tf    # ML library for graphs
import cv2                 # image processing
import numpy as np         # mathematical operations
import os                  # working with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from scipy import io as sio
from tensorflow.python.lib.io import file_io
from io import BytesIO


IMG_SIZE = 80 # our images are 80x80x3
NumClasses = 10

steps = 400
epochs = 300
step_size = 40
total_batch = int(steps/step_size)
LR = 0.00001
#to add additional data for training, change the below path with your newly .mat file
data_location="gs://image-analytics/data/nyu_training_data-400.mat"

def load_data(path):
    matfl = BytesIO(file_io.read_file_to_string(path, binary_mode=True))
    TrainData = sio.loadmat(matfl)
    train_data = TrainData['images']
    names = TrainData['Names']
    train_labels = TrainData['labels']
    
    #Splitting train and CV data
    train = train_data
    cv = train_data[300:]
    
    X = np.zeros((len(train), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    Y = train_labels[0:len(train), :]
    idx = 0
    for i in train:
        img = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
        X[idx, :, :, :] = img
        idx+=1
        
    cv_x = np.zeros((len(cv), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    cv_y = train_labels[300:, :]
    idx = 0
    for i in cv:
        img = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
        cv_x[idx, :, :, :] = img
        idx+=1
    
    test_x = X.copy()
    test_y = Y.copy()
    
    X = X.astype(np.float32)/255.
    Y = Y.astype(np.float32)
    cv_x = cv_x.astype(np.float32)/255.
    cv_y = cv_y.astype(np.float32)
    test_x = test_x.astype(np.float32)/255.
    test_y = test_y.astype(np.float32)

    return X, Y, cv_x, cv_y, test_x, test_y


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


def train_images(path):
    # initialize the graph
    x, y_true, hold_prob1,hold_prob2, hold_prob3, hold_prob4, y_pred = Graph_init()
    
    # get the data
    X, Y, cv_x, cv_y, test_x, test_y = load_data(path)
    
        #Writing Loss and Accuracy
    with tf.name_scope('Loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

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

    #If you are using CPU, just use with tf.Session() as sess:
    #Starting Session

    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter("summary/", graph=tf.get_default_graph())
        for i in range(epochs):
            for j in range(0,steps,step_size): 
                print('epoch', i, 'step', j, X[j:j+step_size, :, :, :].shape, Y[j:j+step_size, :].shape)
                y_pr, tr , c , summary, d = sess.run([y_pred, train,cross_entropy,merged_summary_op,acc],feed_dict={x:X[j:j+step_size, :, :, :] , y_true:Y[j:j+step_size, :], hold_prob1:0.5, hold_prob2:0.5,hold_prob3:0.5,hold_prob4:0.5})
                summary_writer.add_summary(summary, i * total_batch + j)
                acc_train.append(d)
                mean_of_cross_entropy = sess.run(cross_entropy,feed_dict={x:cv_x,y_true:cv_y ,hold_prob1:1.0,hold_prob2:1.0,hold_prob3:1.0,hold_prob4:1.0})
                mean_of_acc = sess.run(acc,feed_dict={x:cv_x ,y_true:cv_y,hold_prob1:1.0,hold_prob2:1.0,hold_prob3:1.0,hold_prob4:1.0})
                cross_entropy_list.append(mean_of_cross_entropy)
                acc_list.append(mean_of_acc)
            print(i,mean_of_cross_entropy,mean_of_acc)
            if i %20 ==0:
                saver.save(sess, "gs://image-analytics/New-model-400/CNN_MC_" + str(i) +".ckpt")
        
        
train_images(data_location)
