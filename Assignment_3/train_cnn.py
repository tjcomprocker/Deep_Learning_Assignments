'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Joshi Tanuj Kirankumar
Roll No.: 16CS60R86

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf
import os


n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


W_conv1 = tf.Variable(tf.random_normal([5,5,1,32]), name = "W_con1")
W_conv2 = tf.Variable(tf.random_normal([5,5,32,64]), name = "W_con2")
W_fc = tf.Variable(tf.random_normal([7*7*64,1024]), name = "W_fc")
W_out = tf.Variable(tf.random_normal([1024, n_classes]), name = "W_out")

b_conv1 = tf.Variable(tf.random_normal([32]), name = "b_conv1")
b_conv2 = tf.Variable(tf.random_normal([64]), name = "b_conv2")
b_fc = tf.Variable(tf.random_normal([1024]), name = "b_fc")
b_out = tf.Variable(tf.random_normal([n_classes]), name = "b_out")

x2 = tf.reshape(x, shape=[-1, 28, 28, 1])

conv1 = tf.nn.relu(conv2d(x2, W_conv1) + b_conv1)
conv1 = maxpool2d(conv1)

conv2 = tf.nn.relu(conv2d(conv1, W_conv2) + b_conv2)
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2,[-1, 7*7*64])
fc = tf.nn.relu(tf.matmul(fc, W_fc)+b_fc)

output = tf.matmul(fc, W_out)+ b_out


def train(trainX, trainY):

    """Reshaping trainX and trainY"""
    
    trainX = trainX.reshape((60000,784)).astype(np.float)
    trainX = trainX / 255
    trainY = trainY.reshape((60000)).astype(np.int)

    """Converting trainY to one hot representation"""
    trY_one_hot = np.zeros((60000, 10))
    trY_one_hot[np.arange(60000), trainY] = 1

    ##############################################################################
    prediction = output

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    all_saver = tf.train.Saver()
    hm_epochs = 2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range (0,60000/batch_size):
                _, c = sess.run([optimizer, cost], feed_dict={x: trainX[i*batch_size:i*batch_size+batch_size-1], y: trY_one_hot[i*batch_size:i*batch_size+batch_size-1]})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        os.system("mkdir model")
        all_saver.save(sess, "model/tj.ckpt")


def test(testX):

    """Reshaping testX"""
    all_saver = tf.train.Saver()

    testX = testX.reshape((10000,784)).astype(np.float)
    testX = testX / 255

    prediction = output

    lab = []

    os.system("mkdir model")
    os.system("wget https://github.com/tjcomprocker/DL-Assignment-3-CNN/blob/master/model/tj.ckpt.meta?raw=true")
    os.system("wget https://github.com/tjcomprocker/DL-Assignment-3-CNN/raw/master/model/checkpoint")
    os.system("wget https://github.com/tjcomprocker/DL-Assignment-3-CNN/blob/master/model/tj.ckpt.index?raw=true")
    os.system("wget https://github.com/tjcomprocker/DL-Assignment-3-CNN/blob/master/model/tj.ckpt.data-00000-of-00001?raw=true")
    os.system("mv checkpoint model/checkpoint")
    os.system("mv tj.ckpt.meta?raw=true model/checkpoint")
    os.system("mv tj.ckpt.index?raw=true model/tj.ckpt.index")
    os.system("mv tj.ckpt.data-00000-of-00001?raw=true model/tj.ckpt.data-00000-of-00001")

    with tf.Session() as sess:
        all_saver.restore(sess, "model/tj.ckpt")        
        predicted = tf.argmax(prediction, 1)
        lab = predicted.eval({x: testX})

        
    return lab