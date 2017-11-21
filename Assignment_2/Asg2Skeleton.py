
# coding: utf-8

# Deep Learning Programming Assignment 2
# --------------------------------------
# Name: Joshi Tanuj Kirankumar
# Roll No.: 16CS60R86
# 
# Submission Instructions:
# 1. Fill your name and roll no in the space provided above.
# 2. Name your folder in format <Roll No>_<First Name>.
#     For example 12CS10001_Rohan
# 3. Submit a zipped format of the file (.zip only).
# 4. Submit all your codes. But do not submit any of your datafiles
# 5. From output files submit only the following 3 files. simOutput.csv, simSummary.csv, analogySolution.csv
# 6. Place the three files in a folder "output", inside the zip.

import gzip
import os
import tensorflow as tf
import numpy as np
import random

## paths to files. Do not change this
simInputFile = "Q1/word-similarity-dataset"
analogyInputFile = "Q1/word-analogy-dataset"
vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
analogyTrainPath = "Q1/wordRep/"
simOutputFile = "Q1/simOutput.csv"
simSummaryFile = "Q1/simSummary.csv"
anaSoln = "Q1/analogySolution.csv"
Q4List = "Q4/wordList.csv"

# Similarity Dataset
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()]
# Analogy dataset
analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

#####################################################################################
#THIS CODE IS ADDED TO ALSO FETCH WORD VECTORS FROM WORDS WHICH ARE IN "wordRep"#####
#####################################################################################
original_examples = len(analogyDataset)
analogy_training_path = "Q1/wordRep/Pairs_from_WordNet/"
for files in os.listdir(analogy_training_path):
        fl = open(analogy_training_path + files)
        for lines in fl:
            lines = lines[:-2]
            analogyDataset.append([lines,lines,lines,lines,'a'])
#####################################################################################

def vectorExtract(simD = simDataset, anaD = analogyDataset, vect = vectorgzipFile):
    simList = [stuff for item in simD for stuff in item]
    analogyList = [thing for item in anaD for stuff in item[0:4] for thing in stuff.split()]
    simList.extend(analogyList)
    wordList = set(simList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]
    
    
    vectorFile.close()
    return wordDict

# Extracting Vectors from Analogy and Similarity Dataset
validateVectors = vectorExtract()

##########THIS WAS ADDED SINCE THE ANALOGY DATASET WILL ACTUALLY SERVE AS VALIDATION SET SO RESETTING IT
##################################################################
analogyDataset = analogyDataset[0:original_examples]
##################################################################

# Dictionary of training pairs for the analogy task
trainDict = dict()
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        trainDict[files] = f

def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):

    """
    Output simSummary.csv in the following format
    Distance Metric, Number of questions which are correct, Total questions evalauted, MRR
    C, 37, 40, 0.61
    """

    """
    Output a CSV file titled "simOutput.csv" with the following columns

    file_line-number, query word, option word i, distance metric(C/E/M), similarity score 

    For the line "rusty | corroded | black | dirty | painted", the outptut will be

    1,rusty,corroded,C,0.7654
    1,rusty,dirty,C,0.8764
    1,rusty,black,C,0.6543


    The order in which rows are entered does not matter and so do Row header names. Please follow the order of columns though.
    """

    w = []
    flag = 0
    i = 1
    count = 0
    c_correct = 0
    e_correct = 0
    m_correct = 0
    c_mrr = float(0)
    e_mrr = float(0)
    m_mrr = float(0)

    # Creating Files
    fl_output = open(simOutputFile,"w")
    fl_summary = open(simSummaryFile,"w")

    #For each line in input file
    for items in inputDS:
        w = []
        c_ls = []
        e_ls = []
        m_ls = []
        flag = 0
        for words in items:
            w.append(vectors.get(words))
    	
        #checking if the word vector exists or not
        for words in w:
            if words == None:
                flag = 1
                break

        # if word vector exists then calculate distance
    	if flag == 0:
            for words in range(1,len(w)):
                
                c = cosine(w[0],w[words])
                e = ecldn(w[0],w[words])
                m = manhattan(w[0],w[words])
                
                fl_output.write(str(i)+","+str(items[0])+","+str(items[words])+",C,"+str(c)+"\n")
                fl_output.write(str(i)+","+str(items[0])+","+str(items[words])+",E,"+str(e)+"\n")
                fl_output.write(str(i)+","+str(items[0])+","+str(items[words])+",M,"+str(m)+"\n")
                
                c_ls.append((words,c))
                e_ls.append((words,e))
                m_ls.append((words,m))
        else:
            i = i + 1
            continue
        
        c_ls = sorted(c_ls, key=lambda x: x[1], reverse = True)
        e_ls = sorted(e_ls, key=lambda x: x[1])
        m_ls = sorted(m_ls, key=lambda x: x[1])

        if c_ls[0][0] == 1:
            c_correct = c_correct + 1
        if e_ls[0][0] == 1:
            e_correct = e_correct + 1
        if m_ls[0][0] == 1:
            m_correct = m_correct + 1

        for index in range(0,len(c_ls)):
            if (c_ls[index][0] == 1):
                c_mrr = c_mrr + float((index+1)**(-1))
                break

        for index in range(0,len(e_ls)):
            if (e_ls[index][0] == 1):
                e_mrr = e_mrr + float((index+1)**(-1))
                break

        for index in range(0,len(m_ls)):
            if (m_ls[index][0] == 1):
                m_mrr = m_mrr + float((index+1)**(-1))
                break

        i = i + 1
        count = count + 1

    #claculating MRR
    c_mrr = c_mrr/count
    e_mrr = e_mrr/count
    m_mrr = m_mrr/count
    
    #writing output in summary file
    fl_summary.write("C, "+str(c_correct)+", "+str(count)+", "+str(c_mrr)+"\n")
    fl_summary.write("E, "+str(e_correct)+", "+str(count)+", "+str(e_mrr)+"\n")
    fl_summary.write("M, "+str(m_correct)+", "+str(count)+", "+str(m_mrr)+"\n")

    fl_summary.close()
    fl_output.close()

#This function calculated cosine similarity between two given vectors and return the similarity value
def cosine(x,y):
	x = map(float, x)
	y = map(float, y)
	x = np.asarray(x)
	y = np.asarray(y)
	return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

#This function calculated euclidean distance between two given vectors and returns the distance value
def ecldn(x,y):
	x = map(float, x)
	y = map(float, y)
	x = np.asarray(x)
	y = np.asarray(y)
	return np.sqrt(np.sum(np.power((x-y),2)))

#This function calculated manhattan distance between two given vectors and returns the distance value
def manhattan(x,y):
	x = map(float, x)
	y = map(float, y)
	x = np.asarray(x)
	y = np.asarray(y)
	return np.sum(np.absolute(x-y))

#Declaring variables for the neural network
h1_nodes = 500
h2_nodes = 500
h3_nodes = 500
classes = 5
input_size = 300*12
k_fold = 5
k_fold_accuracy = 0

x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float')

#This function defines the abstract model of the neural network
def model(input):

    #weights of each layer
    h1 = {'weights':tf.Variable(tf.random_normal([input_size, h1_nodes])) , 'biases':tf.Variable(tf.random_normal([h1_nodes]))}
    h2 = {'weights':tf.Variable(tf.random_normal([h1_nodes, h2_nodes])) , 'biases':tf.Variable(tf.random_normal([h2_nodes]))}
    h3 = {'weights':tf.Variable(tf.random_normal([h2_nodes, h3_nodes])) , 'biases':tf.Variable(tf.random_normal([h3_nodes]))}
    o = {'weights':tf.Variable(tf.random_normal([h3_nodes, classes])) , 'biases':tf.Variable(tf.random_normal([classes]))}

    #functions in each layer
    layer1 = tf.add(tf.matmul(input,h1['weights']) , h1['biases'])
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1,h2['weights']) , h2['biases'])
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.add(tf.matmul(layer2,h3['weights']) , h3['biases'])
    layer3 = tf.nn.relu(layer3)
    output = tf.matmul(layer3,o['weights']) + o['biases']

    return output

#This function takes the model and trains on the training set created in analogyTask function
def train_analogy(x,training_dataset,labels,test_dataset,test_labels):
    prediction = model(x)

    remaining = test_dataset.shape[0] % k_fold
    flag = 0
    last = 0
    last_labels = 0
    k_fold_accuracy = 0
    if remaining != 0:
        last = test_dataset[-remaining:]
        test_dataset = test_dataset[:-remaining]
        last_labels = test_labels[-remaining:]
        test_labels = test_labels[:-remaining]
        flag = 1

    chunks = np.vsplit(test_dataset,k_fold)
    chunks_labels = np.vsplit(test_labels,k_fold)
    if flag == 1:
        chunks.append(last)
        chunks_labels.append(last_labels)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction , labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    iterations = 5
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            iteration_loss = 0
            iteration_x = training_dataset
            iteration_y = labels
            _ , iteration_cost = sess.run([optimizer, cost] , feed_dict={x: iteration_x , y: iteration_y})
            iteration_loss = iteration_loss + iteration_cost

        equals = tf.equal(tf.argmax(prediction , 1) , tf.argmax(y , 1))
        accuracy = tf.reduce_mean(tf.cast(equals , 'float'))

        for j in range(0,k_fold):
            iteration_loss = 0
            iteration_x = chunks.pop(0)
            iteration_y = chunks_labels.pop(0)
            iteration_x2 = np.empty(shape=(0,0))
            iteration_y2 = np.empty(shape=(0,0))

            iteration_x2 = np.concatenate((chunks[0], chunks[1]), axis=0)
            iteration_y2 = np.concatenate((chunks_labels[0], chunks_labels[1]), axis=0)
            
            for batman in range(2,len(chunks)):
                iteration_x2 = np.concatenate((iteration_x2, chunks[batman]), axis=0)
                iteration_y2 = np.concatenate((iteration_y2, chunks_labels[batman]), axis=0)

            _ , iteration_cost = sess.run([optimizer, cost] , feed_dict={x: iteration_x2 , y: iteration_y2})
            iteration_loss = iteration_loss + iteration_cost
            chunks.append(iteration_x)
            chunks_labels.append(iteration_y)
            k_fold_accuracy = k_fold_accuracy + accuracy.eval({x:iteration_x , y:iteration_y})
       
    return k_fold_accuracy

def analogyTask(inputDS=analogyDataset,outputFile = anaSoln ): # add more arguments if required
    
    """
    Output a file, analogySolution.csv with the following entris
    Query word pair, Correct option, predicted option    
    """

    #Creating the training set from wordRep
    accuracy = 0
    validation_set = inputDS
    analogy_training_path = "Q1/wordRep/Pairs_from_WordNet/"

    ls = []
    test_dataset = []
    test_labels = []
    validation_set2 =[]
    flag = 0

    for items in validation_set:
        flag = 0
        for i in range(0,6):
            temp = items[i].split(" ")
            if validateVectors.get(temp[0]) == None or validateVectors.get(temp[1]) == None:
                flag = 1
        if flag == 0:
            validation_set2.append(items) 
        
    for items in validation_set2:
        for i in range(0,6):
            temp = items[i].split(" ")
            for values in validateVectors.get(temp[0]):
                ls.append(values)
            for values in validateVectors.get(temp[1]):
                ls.append(values)

        if items[6] == 'a':
            test_labels.append(0)
        elif items[6] == 'b':
            test_labels.append(1)
        elif items[6] == 'c':
            test_labels.append(2)
        elif items[6] == 'd':
            test_labels.append(3)
        elif items[6] == 'e':
            test_labels.append(4)

    test_dataset = np.reshape(np.asarray(ls),(len(ls)/(300*12),300*12))
    a = np.asarray(test_labels)
    b = np.zeros((len(test_labels), 5))
    b[np.arange((len(b))), a] = 1
    test_labels = b
     
    files_data = {}
    for files in os.listdir(analogy_training_path):
        fl = open(analogy_training_path + files)
        for lines in fl:
            lines = lines[:-2].split("\t")
            if files_data.get(files) == None:
                if validateVectors.get(lines[0]) != None and validateVectors.get(lines[1]) != None:
                    files_data[files] = [map(lambda x:x.lower(),lines)]
            else:
                if validateVectors.get(lines[0]) != None and validateVectors.get(lines[1]) != None:
                    files_data[files].append(map(lambda x:x.lower(),lines))


    training_dataset = []
    random_index  = 0
    
    
    for keys in files_data.keys():
        for items in files_data[keys]:
            ls = []
            random_index = random.randint(0,len(files_data[keys])-1)
            
            while files_data[keys].index(items) == random_index:
                random_index = random.randint(0,len(files_data[keys])-1)
                pass

            random_index = files_data[keys][random_index]
            ls.append(random_index)
            
            temp2 = []

            for keys2 in files_data.keys():
                if keys != keys2:
                    temp2 = temp2 + files_data[keys2]

            ls.append(temp2[random.randint(0,len(temp2)-1)])
            ls.append(temp2[random.randint(0,len(temp2)-1)])
            ls.append(temp2[random.randint(0,len(temp2)-1)])
            ls.append(temp2[random.randint(0,len(temp2)-1)])

            random.shuffle(ls)

            ls.append(ls.index(random_index))

            training_dataset.append(items)
            for items2 in ls:
                training_dataset.append(items2)

    ls = []
    labels = []
    for items in training_dataset:
        if isinstance(items, list):
            temp = validateVectors.get(items[0])
            for values in temp:
                ls.append(values)

            temp = validateVectors.get(items[1])
            for values in temp:
                ls.append(values)
        else:
            labels.append(items)
    
    training_dataset = np.reshape(np.asarray(ls),(len(ls)/(300*12),300*12))
    a = np.asarray(labels)
    b = np.zeros((len(labels), 5))
    b[np.arange((len(b))), a] = 1
    labels = b

    accuracy = train_analogy(x,training_dataset,labels,test_dataset,test_labels)
   
    return accuracy #return the accuracy of your model after 5 fold cross validation


# In[60]:

def derivedWordTask(inputFile = Q4List):

    cosVal1 = 0
    cosVal2 = 0

    """
    Output vectors of 3 files:
    1)AnsFastText.txt - fastText vectors of derived words in wordList.csv
    2)AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv
    3)AnsModel.txt - Vectors for derived words as provided by the model
    
    For all the three files, each line should contain a derived word and its vector, exactly like 
    the format followed in "glove.6B.300d.txt"
    
    word<space>dim1<space>dim2........<space>dimN
    charitably 256.238 0.875 ...... 1.234
    
    """
    
    """
    The function should return 2 values
    1) Averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.
    
        - if there are 3 derived words in wordList.csv, say word1, word2, word3
        then find the cosine similiryt between word1 in AnsFastText.txt and word1 in AnsModel.txt.
        - Repeat the same for word2 and word3.
        - Average the 3 cosine similarity values
        - DO the same for word1 to word3 between the files AnsLzaridou.txt and AnsModel.txt 
        and average the cosine simialities for valuse so obtained
        
    """
    return cosVal1,cosVal2
   


# In[ ]:

def main():
    similarityTask()
    anaSim = analogyTask()
    derCos1,derCos2 = derivedWordTask()
    #derivedWordTask()

if __name__ == '__main__':
    main()
