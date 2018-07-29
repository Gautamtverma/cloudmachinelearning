import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import re
from random import randint
import datetime
import tarfile
import argparse
from tensorflow.python.lib.io import file_io
from io import BytesIO


maxSeqLength = 250  # Maximum length of sentence
numDimensions = 300  # Dimensions for each word vector
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 5001
output_checkpoint_cnt=1000 # less than or equal to iterations
summary_out_cnt=1000 # less than or equal to iterations

wordVectorsLoc = 'gs://sent-data/trainingdata/wordVectors.npy'
#idsMatrixLoc = 'gs://sent-data/newTrain/idsMatrix-1000.npy'
idsMatrixLoc = 'gs://sent-data/newTrain/idsMatrix-24500.npy'

# To run 1000 reviews uncomment below variables
#tr_st = 1
#tr_sp = 399
#tr_e_st = 599
#tr_e_sp = 999
#ts_st = 499

# To run 24500 reviews uncomment below variables
tr_st = 1
tr_sp = 11499
tr_e_st = 13499
tr_e_sp = 24499
ts_st = 12499



def getTrainBatch(ids):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(tr_st,tr_sp)
            labels.append([1,0])
        else:
            num = randint(tr_e_st,tr_e_sp)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch(ids):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(tr_sp,tr_e_st)
        if (num <=ts_st):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


def sentiment_analysis_training():
    wv = BytesIO(file_io.read_file_to_string(wordVectorsLoc, binary_mode=True))
    id_data = BytesIO(file_io.read_file_to_string(idsMatrixLoc, binary_mode=True))
    wordVectors = np.load(wv)
    print ('Loaded the word vectors!')
    ids = np.load(id_data)

    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    logdir = "gs://sent-data/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch(ids);
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if (i % summary_out_cnt == 0):
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if (i % output_checkpoint_cnt == 0 and i != 0):
            save_path = saver.save(sess, "gs://sent-data/model-5000/trained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

    writer.close()
    
sentiment_analysis_training()

#if __name__ == '__main__':
#    # Accessing the file names from command line
#    parser = argparse.ArgumentParser(description=__doc__,
#                                     formatter_class=argparse.RawDescriptionHelpFormatter)
#    # Adding arguments for Directory names and help information about Directory
#    parser.add_argument('user_first_arg',help='The positive Folder of the review you\'d like to analyze.')
#    parser.add_argument('user_second_arg', help='The negative Folder you need to of the review you\'d like to analyze.')
#
#    args = parser.parse_args()
#    # Calling the sentiment_detect function with parameters.
#    sentiment_analysis_training(args.user_first_arg, args.user_second_arg)

