import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import numpy as np
from scipy import io as sio

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = 'predictions'
IMG_SIZE = 80
NumClasses = 96

LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    # Input Layer
    input_layer = tf.reshape(features[INPUT_TENSOR_NAME], [-1, IMG_SIZE, IMG_SIZE, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
    

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 10 * 10 * 64])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == Modes.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=NumClasses)

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = logits
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.float32)
        loss = tf.losses.mean_squared_error(labels, logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        y_pred_out = tf.greater(logits, 0.5)
        y_pred_out = tf.cast(y_pred_out, tf.float32)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, y_pred_out)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def read_and_decode(filename):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename)
    
    TrainData = sio.loadmat(filename)
    train_data = TrainData['images']
    names = TrainData['Names']
    train_labels = TrainData['labels']
    
    #Splitting train and CV data
    train = train_data[:50]
    cv = train_data[50:]
    
    image = np.zeros((len(train), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    label = train_labels[0:len(train), :]
    idx = 0
    for i in train:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        img[:, :, 0] = np.resize(i[:, :, 0], (IMG_SIZE, IMG_SIZE))
        img[:, :, 1] = np.resize(i[:, :, 1], (IMG_SIZE, IMG_SIZE))
        img[:, :, 2] = np.resize(i[:, :, 2], (IMG_SIZE, IMG_SIZE))
        image[idx, :, :, :] = img
        idx+=1
        
#     cv_x = np.zeros((len(cv), IMG_SIZE, IMG_SIZE, 3), np.uint8)
#     cv_y = train_labels[len(train):, :]
#     idx = 0
#     for i in cv:
#         img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
#         img[:, :, 0] = np.resize(i[:, :, 0], (IMG_SIZE, IMG_SIZE))
#         img[:, :, 1] = np.resize(i[:, :, 1], (IMG_SIZE, IMG_SIZE))
#         img[:, :, 2] = np.resize(i[:, :, 2], (IMG_SIZE, IMG_SIZE))
        
#         cv_x[idx, :, :, :] = img
#         idx+=1
    
#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             'image_raw': tf.FixedLenFeature([], tf.string),
#             'label': tf.FixedLenFeature([], tf.int64),
#         })
    features={
            'image_raw': tf.FixedLenFeature([], tf.float32),
            #'label': tf.FixedLenFeature([], tf.int64),
        }
#     image = tf.decode_raw(features['image_raw'], tf.uint8)
#     image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    image = tf.cast(image, tf.float32) * (1. / 255.)
    label = tf.cast(label, tf.float32)

    return image, label


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'nyu_training_data.mat', batch_size=50)


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'nyu_training_data.mat', batch_size=50)


def _input_fn(training_dir, training_filename, batch_size=50):
#     test_file = os.path.join(training_dir, training_filename)
#     filename_queue = tf.train.string_input_producer([test_file])
    file_name = os.path.join(training_dir, training_filename)
    image, label = read_and_decode(file_name)
    #     images, labels = tf.train.batch(
    #         [image, label], batch_size=batch_size,
    #         capacity=1000 + 3 * batch_size)

    return {INPUT_TENSOR_NAME: image}, label
