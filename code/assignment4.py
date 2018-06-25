import sys
import tensorflow as tf
import random
import numpy as np
from numpy import array

def readfile(filename):
    array=[]
    with open(filename, encoding="utf8") as f:
        for line in f:
            if len(line) > 1:
                line = line[0:len(line) - 1]
                array.append(line)
    return array


def create_weight_bias(input,hidden_one,hidden_two,number_of_classes):
    weight = {
        'h1': tf.Variable(tf.random_normal([input, hidden_one])),
        'h2': tf.Variable(tf.random_normal([hidden_one, hidden_two])),
        'out': tf.Variable(tf.random_normal([hidden_two, number_of_classes]))
    }
    bias = {
        'b1': tf.Variable(tf.random_normal([hidden_one])),
        'b2': tf.Variable(tf.random_normal([hidden_two])),
        'out': tf.Variable(tf.random_normal([number_of_classes]))
    }
    return weight,bias


def split_vector(array_vector):
    word=[]
    for i in range(len(array_vector)):
        result=array_vector[i].split(" ")
        for j in range(len(result[0])):
            if result[0][j]==":":
                word.append(result[0][:j])
                result[0]=result[0][j+1:]
                array_vector[i]=result
                break

    return word,array_vector



def multilayer_perceptron(x,weight,bias):
    layer_1 = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weight['h2']), bias['b2'])
    out_layer = tf.matmul(layer_2, weight['out']) + bias['out']
    return out_layer

def create_data(data,train_length):
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    temp1 = [1.0, 0.0]
    temp2 = [0.0, 1.0]
    for i in range(len(data)):
        result = data[i].split("-:")
        if i<train_length:
            train_x.append(result[0])
            if(result[1]=="[1.0, 0.0]"):
                train_y.append(temp1)
            else:
                train_y.append(temp2)

        elif i>=train_length:
            test_x.append(result[0])
            if (result[1] == "[1.0, 0.0]"):
                test_y.append(temp1)
            else:
                test_y.append(temp2)


    return train_x,train_y,test_x,test_y


def calculate_sentence(train_x,test_x,vector,word):

    for i in range(len(train_x)):
        result = train_x[i].split(" ")
        after = np.zeros(200, dtype=float)
        for j in range(len(result)):
            try:
                index=word.index(result[j])
                temp = np.array(vector[index], dtype=float)
                after = after + temp
            except:
                index=-1

        train_x[i]=after


    for i in range(len(test_x)):
        result = test_x[i].split(" ")
        after = np.zeros(200, dtype=float)
        for j in range(len(result)):
            try:
                index=word.index(result[j])
                temp = np.array(vector[index], dtype=float)
                after = after + temp
            except:
                index=-1

        test_x[i]=after

    return train_x,test_x


def main():

    array_positives=readfile(sys.argv[1])
    array_negatives = readfile(sys.argv[2])
    array_vector=readfile(sys.argv[3])
    word,array_vector=split_vector(array_vector)
    temp1=[1.0 ,0.0]
    temp2=[0.0 ,1.0]
    for i in range(len(array_positives)):
        array_positives[i]=array_positives[i]+ "-:"+ str(temp1)

    for i in range(len(array_negatives)):
        array_negatives[i] = array_negatives[i] + "-:" + str(temp2)

    for i in range(len(array_negatives)):
        array_positives.append(array_negatives[i])

    random.shuffle(array_positives)
    train_length=int(int(sys.argv[4])*len(array_positives)/100)
    train_x,train_y,test_x,test_y=create_data(array_positives,train_length)
    train_x,test_x=calculate_sentence(train_x,test_x,array_vector,word)

    hidden_one = 100
    hidden_two=100
    input=200
    number_of_classes=2
    x = tf.placeholder("float", [None, input])
    y = tf.placeholder("float", [None, number_of_classes])
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 50

    weight,bias=create_weight_bias(input,hidden_one,hidden_two,number_of_classes)
    logits = multilayer_perceptron(x,weight,bias)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(train_x) / batch_size)
            for i in range(total_batch):
                _, c = sess.run([train_op, loss], feed_dict={x: train_x, y: train_y})
                avg_cost += c / total_batch


        pred = tf.nn.relu(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))



main()