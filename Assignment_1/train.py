'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Joshi Tanuj Kirankumar
Roll No.: 16CS60R86

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np

""" Sigmoid function"""
def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-1.0 * x))
    return s 


"""Softmax function"""
def softmax(x):
    x_max = np.max(x, axis=0)
    r = np.exp(x - x_max)
    r = r / np.sum(r, axis=0)
    return r


"""weights being randomy initialized"""
w1 = np.random.uniform(-1/10**0.5, 1/10**0.5, (784, 10))
w2 = np.random.uniform(-1/10**0.5, 1/10**0.5, (10, 10))


def train(trainX, trainY):
    
    """Reshaping trainX and trainY"""
    trainX = trainX.reshape((60000,784,1)).astype(np.float)
    trainY = trainY.reshape((60000)).astype(np.int)

    """Converting trainY to one hot representation"""
    trY_one_hot = np.zeros((60000, 10,1))
    trY_one_hot[np.arange(60000), trainY] = 1

    """setting eta value to 0.2"""
    eta = 0.2

    """Initializing ouput of 1st and secons layer"""
    z1 = []
    z2 = []

    """Forward Pass"""
    for i in range(0,60000):
        z1.append(sigmoid(np.dot(w1.T , trainX[i])))

    z1 = np.reshape(z1,(60000,10,1)).astype(np.float)

    for i in range(0,60000):
        z2.append(softmax(np.dot(w2.T , z1[i])))

    z2 = np.reshape(z2,(60000,10,1)).astype(np.float)

    """Back Propagation"""
    for iterations in range (0,1):
        for example in range(0,10000):
            error_layer_1 = 0
            for i in range(0,10):
                for j in range (0,10):
                    temp  = (z2[example][i] - trY_one_hot[example][i]) * z1[example][j]
                    error_layer_1 = error_layer_1 + w2[j][i] * temp * (1 - z1[example][j])
                    w2[j][i] = w2[j][i] - eta * temp

                for k in range (0,784):
                    w1[k][j] = w1[k][j] - eta * error_layer_1 * trainX[example][k]

    """Writing Generated files in weights.txt file"""            
    w1_file = open("weights1.txt","w+")
    w2_file = open("weights2.txt","w+")

    for i in range (0,784):
        for j in range (0,10):
            w1_file.write(str(w1[i][j])+'\n')
    
    for i in range (0,10):
        for j in range (0,10):
            w2_file.write(str(w2[i][j])+'\n')
    
    w1_file.close()
    w2_file.close()

    

def test(testX):
    """Reshaping testX"""
    testX = testX.reshape((10000,784,1)).astype(np.float)

    testY = []
    z1 = []
    z2 = []

    count = 0
    w1 = []
    w2 = []

    with open("weights1.txt") as fl:
        for lines in fl:
            w1.append(float(lines))
    
    with open("weights2.txt") as fl:
        for lines in fl:
            w2.append(float(lines))
        
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    w1 = w1.reshape((784,10)).astype(np.float)
    w2 = w2.reshape((10,10)).astype(np.float)

    """Generating lables"""
    for i in range(0,10000):
        z1.append(sigmoid(np.dot(w1.T , testX[i])))

    z1 = np.reshape(z1,(10000,10,1)).astype(np.float)

    for i in range(0,10000):
        z2.append(softmax(np.dot(w2.T , z1[i])))

    z2 = np.reshape(z2,(10000,10,1)).astype(np.float)

    for i in range(0,10000):
        testY.append(np.argmax(z2[i], axis=0))

    testY = np.asarray(testY)
    testY = testY.reshape((10000,)).astype(np.int)
    testY = np.asarray(testY)
    
    return testY
