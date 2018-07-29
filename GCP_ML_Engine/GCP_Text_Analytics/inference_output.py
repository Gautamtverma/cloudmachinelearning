import os, io, sys
import tensorflow as tf
import re
import numpy as np, csv

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="apikey.json"

numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 10

def cleanSentences(string, strip_special_chars):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence, strip_special_chars, wordsList):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence, strip_special_chars)
    split = cleanedSentence.split()
    if len(split) >=250 :
        return sentenceMatrix, 0
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix, 1

def inference_output(dirname, Actual_sentiment,writer):
    wordsList = np.load('wordsList.npy').tolist()
    wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
    wordVectors = np.load('wordVectors.npy')

    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('gs://sent-data/model-5000/'))

    listdir = os.listdir(dirname)

    # picking a random file name in directory
    for filename in listdir:
        review_filename = dirname + '/' + filename
        # Reading the content of file
        with io.open(review_filename, 'rb') as txt_file:
            inputText = txt_file.read()

        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

        inputMatrix, flag = getSentenceMatrix(inputText, strip_special_chars, wordsList)

        if flag == 0:
            continue

        predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
        # predictedSentiment[0] represents output score for positive sentiment
        # predictedSentiment[1] represents output score for negative sentiment
        print filename
        if (predictedSentiment[0] > predictedSentiment[1]):
            #print "Positive Sentiment"
            #print "Score : ", predictedSentiment[0]
            writer.writerow({'FileName': filename, 'SentimentScore': predictedSentiment[0],
                             'PredictedSentiment': 'POSITIVE', 'ActualSentiment': Actual_sentiment})
        else:
            #print "Negative Sentiment"
            #print "Score : ", predictedSentiment[1]
            writer.writerow({'FileName': filename, 'SentimentScore': predictedSentiment[1],
                             'PredictedSentiment': 'NEGATIVE', 'ActualSentiment': Actual_sentiment})
