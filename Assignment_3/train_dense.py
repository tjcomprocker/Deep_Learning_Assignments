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

h1_nodes = 500
h2_nodes = 500
h3_nodes = 500
classes = 10
input_size = 784

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float')

wh1 = tf.Variable(tf.random_normal([input_size, h1_nodes]), name ="wh1")
bh1 = tf.Variable(tf.random_normal([h1_nodes]), name ="bh1")
wh2 = tf.Variable(tf.random_normal([h1_nodes, h2_nodes]), name ="wh2")
bh2 = tf.Variable(tf.random_normal([h2_nodes]), name ="bh2")
wh3 = tf.Variable(tf.random_normal([h2_nodes, h3_nodes]), name ="wh3")
bh3 = tf.Variable(tf.random_normal([h3_nodes]), name ="bh3")
wo = tf.Variable(tf.random_normal([h3_nodes, classes]), name ="wo")
bo = tf.Variable(tf.random_normal([classes]), name ="bo")

#functions in each layer
layer1 = tf.add(tf.matmul(x,wh1) , bh1)
layer1 = tf.nn.relu(layer1)
layer2 = tf.add(tf.matmul(layer1,wh2) , bh2)
layer2 = tf.nn.relu(layer2)
layer3 = tf.add(tf.matmul(layer2,wh3) , bh3)
layer3 = tf.nn.relu(layer3)
output = tf.matmul(layer3,wo) + bo

def train(trainX, trainY):

	"""Reshaping trainX and trainY"""

	trainX = trainX.reshape((60000,784)).astype(np.float)
	trainX = trainX / 255
	trainY = trainY.reshape((60000)).astype(np.int)

	"""Converting trainY to one hot representation"""
	trY_one_hot = np.zeros((60000,10))
	trY_one_hot[np.arange(60000), trainY] = 1

	prediction = output

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))
	optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
	saver = tf.train.Saver()
	iterations = 100

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		for i in range(iterations):
			iteration_loss = 0
			_ , iteration_cost = sess.run([optimizer, cost] , feed_dict={x: trainX , y: trY_one_hot})
			iteration_loss = iteration_loss + iteration_cost
			print ("loss in "+str(i)+" = "+str(iteration_loss))

		os.system("mkdir model2")
		saver.save(sess, "model2/tj.ckpt")

def test(testX):
    """Reshaping testX"""
    saver = tf.train.Saver()
    testX = testX.reshape((10000,784)).astype(np.float)
    testX = testX / 255
    lab = []

    os.system("mkdir model2")
    os.system("wget https://github.com/tjcomprocker/Deep-Learning-Assignment-3/blob/master/tj.ckpt.meta?raw=true")
    os.system("wget https://github.com/tjcomprocker/Deep-Learning-Assignment-3/raw/master/checkpoint")
    os.system("wget https://github.com/tjcomprocker/Deep-Learning-Assignment-3/blob/master/tj.ckpt.index?raw=true")
    os.system("wget https://github.com/tjcomprocker/Deep-Learning-Assignment-3/blob/master/tj.ckpt.data-00000-of-00001?raw=true")
    os.system("mv checkpoint model2/checkpoint")
    os.system("mv tj.ckpt.meta?raw=true model2/checkpoint")
    os.system("mv tj.ckpt.index?raw=true model2/tj.ckpt.index")
    os.system("mv tj.ckpt.data-00000-of-00001?raw=true model2/tj.ckpt.data-00000-of-00001")

    with tf.Session() as sess:
        saver.restore(sess, "model2/tj.ckpt")
        predicted = tf.argmax(output, 1)        
        lab = predicted.eval({x: testX})
    
    return lab
