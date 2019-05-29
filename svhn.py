#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
from scipy import io
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import os
import gzip
import urllib.request


# In[ ]:


epochs_number = 5
batch_size = 256

categories_number = 10 
conv_depth_1 = 32 
conv_depth_2 = 32
conv_depth_3 = 64 
conv_depth_4 = 64

n_full = 32 #number of neurons in the last fully connected layer

max_pool_width_1 = max_pool_height_1 = 16 #size after 1st max pool
max_pool_width_2 = max_pool_height_2 = 8 #size after 2nd pool


# In[34]:


def convNetwork(x, trainingPhase, dropoutRateConv, dropoutRateFull):
    
    print('shape of x:'+str(x.get_shape()))
    print('Current phase: '+str(trainingPhase))
    # make input 2D before convolution 
    image_resized = tf.reshape(x, [-1,32,32,3])
        
    print('l1: input shape 1: '+str(image_resized.get_shape()))

    grey_image = tf.image.rgb_to_grayscale(image_resized)
    
    #1st convolution
    conv1 = tf.layers.conv2d(grey_image, filters=32, kernel_size=[5,5], strides=[1, 1], padding='SAME',activation=tf.nn.relu)

    print('l2: input shape 1: '+str(conv1.get_shape()))

    batchNorm1 = tf.layers.batch_normalization(conv1, 
                                               center=True, scale=True, 
                                          training=trainingPhase)

    
    #2nd convolution

    conv2 = tf.layers.conv2d(batchNorm1,filters=32, kernel_size=[5,5], strides=[1, 1], padding='SAME',activation=tf.nn.relu)

    batchNorm2 = tf.layers.batch_normalization(conv2, center=True, scale=True, training=trainingPhase)

    #Most house numbers are white on dark - therefore max pooling is used.
    #https://www.researchgate.net/figure/Toy-example-illustrating-the-drawbacks-of-max-pooling-and-average-pooling_fig2_300020038
    
    
    maxPool1 = tf.layers.max_pooling2d(batchNorm2, pool_size=[2, 2], strides=2, padding='SAME')

    
    print('after max pool 1: '+str(maxPool1.get_shape()))


    dropoutLayer1 = tf.layers.dropout(maxPool1, rate=dropoutRateConv)
    
    
    
    
    
    
    print('conv3: input shape: '+str(dropoutLayer1.get_shape()))

    #3rd convolution
    conv3 = tf.layers.conv2d(dropoutLayer1,filters=64, kernel_size=[5,5], strides=[1, 1], padding='SAME',activation=tf.nn.relu)

    batchNorm3 = tf.layers.batch_normalization(conv3, center=True, scale=True, training=trainingPhase)

    
    #4th convolution

    conv4 = tf.layers.conv2d(batchNorm3,filters=64, kernel_size=[5,5], strides=[1, 1], padding='SAME',activation=tf.nn.relu)

    batchNorm4 = tf.layers.batch_normalization(conv4, center=True, scale=True, training=trainingPhase)


    #Most house numbers are white on dark - therefore max pooling is used.
    #https://www.researchgate.net/figure/Toy-example-illustrating-the-drawbacks-of-max-pooling-and-average-pooling_fig2_300020038
        
    maxPool2 = tf.layers.max_pooling2d(batchNorm4, pool_size=[2, 2], strides=2, padding='SAME')

    
    print('after max pool 2: '+str(maxPool2.get_shape()))
    
    dropoutLayer2 = tf.layers.dropout(maxPool2, rate=dropoutRateConv)
    
    
    

    
    print('flatten: input shape: '+str(dropoutLayer2.get_shape()))
    
    # Flatten our 2D tensor back to 1D so that we can feed it into a fully connected layer
    h_pool_flat = tf.reshape(dropoutLayer2, [-1, max_pool_width_2*max_pool_height_2*conv_depth_4])
    
    # Fully connected layer with RELU activation
    print('full1: input shape: '+str(h_pool_flat.get_shape()))

    full = tf.layers.dense(inputs=h_pool_flat, units=n_full, activation=tf.nn.relu)
    print('full output shape: '+str(full.get_shape()))

    dropoutLayer3 = tf.layers.dropout(full, rate=dropoutRateFull)
    
    # Output layer with softmax activation

    outLayer = tf.layers.dense(inputs=dropoutLayer3, units=10)
    #outLayer = tf.nn.softmax(outLayer)
    print('logits shape: '+str(outLayer.get_shape()))

    
    return outLayer
    


# In[36]:


def traintest():
    trainingSetURL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    testSetURL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    
    responseTraining = urllib.request.urlopen(trainingSetURL)
    with open("train_32x32.mat", 'wb') as file:
        file.write(responseTraining.read())
        
    responseTest = urllib.request.urlopen(testSetURL)
    with open("test_32x32.mat", 'wb') as file:
        file.write(responseTest.read())
    
    trainingFile = io.loadmat('train_32x32.mat') 
    testFile = io.loadmat('test_32x32.mat') 
    X_trainRGB = trainingFile['X']
    X_testRGB = testFile['X']
    y_train = trainingFile['y']
    y_test = testFile['y']

    X_testRGBTransposed=X_testRGB.transpose(3,0,1,2)
    X_trainRGBTransposed=X_trainRGB.transpose(3,0,1,2)

    X_testRGBTransposed=X_testRGBTransposed.astype('float32')
    X_trainRGBTransposed=X_trainRGBTransposed.astype('float32')

    X_testRGBTransposed -= np.mean(X_testRGBTransposed, axis = 0)

    X_trainRGBTransposed -= np.mean(X_trainRGBTransposed, axis = 0)

    y_test=y_test.reshape(len(y_test))
    y_train=y_train.reshape(len(y_train))

    percentageOfTrainingSet = 1

    percentageOfTestSet = 1

    X_train=X_trainRGBTransposed
    X_test=X_testRGBTransposed

    X_train = X_train[0:int(len(X_train)*percentageOfTrainingSet)]
    X_test = X_test[0:int(len(X_test)*percentageOfTestSet)]
    y_train = y_train[0:int(len(y_train)*percentageOfTrainingSet)]
    y_test = y_test[0:int(len(y_test)*percentageOfTestSet)]


    X_test1=X_test[np.where(y_test == 1)]
    X_test2=X_test[np.where(y_test == 2)]
    X_test3=X_test[np.where(y_test == 3)]
    X_test4=X_test[np.where(y_test == 4)]
    X_test5=X_test[np.where(y_test == 5)]
    X_test6=X_test[np.where(y_test == 6)]
    X_test7=X_test[np.where(y_test == 7)]
    X_test8=X_test[np.where(y_test == 8)]
    X_test9=X_test[np.where(y_test == 9)]
    X_test0=X_test[np.where(y_test == 10)]
    
    tf.reset_default_graph()

    y_train_encoded = tf.one_hot(indices=y_train, depth=10)
    y_test_encoded = tf.one_hot(indices=y_test, depth=10)
    
    tf.GraphKeys.USEFUL = 'useful'

    # tf Graph input
    x = tf.placeholder("float", [None, 32, 32, 3], name='x_placeholder')
    y = tf.placeholder("float", [None, categories_number], name='y_placeholder')
    dropoutRateConv = tf.placeholder_with_default(1.0, shape=(), name='dropoutRateConv')
    dropoutRateFull = tf.placeholder_with_default(1.0, shape=(), name='dropoutRateFull')

    global_step = tf.Variable(0, trainable=False)
    learning_rate = 0.0001

    trainingPhase = tf.placeholder_with_default(False, (), 'trainingPhase')

    predictions = convNetwork(x, trainingPhase, dropoutRateConv, dropoutRateFull)
    
    tf.add_to_collection(tf.GraphKeys.USEFUL, x)
    tf.add_to_collection(tf.GraphKeys.USEFUL, dropoutRateConv)
    tf.add_to_collection(tf.GraphKeys.USEFUL, dropoutRateFull)
    tf.add_to_collection(tf.GraphKeys.USEFUL, trainingPhase)

    #misclassification rate array
    misclassificationRates = []

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        print('shape of predictions: '+str(predictions))
        print('shape of labels: '+str(y))

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        #cost

    # initializing the tf variables
    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        print(sess.run(init))

        y_train=y_train_encoded.eval()
        y_test=y_test_encoded.eval()
        previousMiscRate = 1
        # training 
        for epoch in range(epochs_number):
            print("Epoch " + str(epoch + 1) + " of " + str(epochs_number), end=" ")        
            avg_cost = 0.
            total_batch = int(len(X_train)/batch_size)
            for i in range(total_batch):
                batch_x = X_train[i*batch_size:min((i+1)*batch_size,len(X_train))].astype('float32')
                batch_y = y_train[i*batch_size:min((i+1)*batch_size,len(y_train))]
                #print('Placeholder X and Y shape: '+str(np.shape(x))+", "+str(np.shape(y)))
                #print('type of batch x: '+str(type(batch_x)))
                #print('dtype of batch x: '+str(batch_x.dtype))

                #print('Batch X and Y shape: '+str(np.shape(batch_x))+", "+str(np.shape(batch_y)))
                #print('Batch X and Y length: '+str(len(batch_x))+", "+str(len(batch_y)))
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, trainingPhase:True, dropoutRateConv:0.7, dropoutRateFull: 0.1})
                print('Cost: '+str(c))
                avg_cost += c / total_batch
                #print('Cost:' + str(c))
            # test model
            predictionTensor=tf.argmax(predictions, 1, name='predictionTensor')
            tf.add_to_collection(tf.GraphKeys.USEFUL, predictionTensor)
            predictionEncoded = tf.one_hot(indices=predictionTensor, depth=10)

            correctPredictions = tf.equal(predictionTensor, tf.argmax(y, 1))
            # accuracy
            accuracy = tf.reduce_mean(tf.cast(correctPredictions, "float")) 
            print('Test X and Y shape: '+str(np.shape(X_test))+", "+str(np.shape(y_test)))

            misclassificationRate=(1 - accuracy.eval({x: X_test, y: y_test, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0}))

            if misclassificationRate>previousMiscRate:
                print('Early stopping.')
                break
            previousMiscRate=misclassificationRate
            print('Error Rate: '+str(misclassificationRate))
            misclassificationRates.append(misclassificationRate)

        print("\r",end="")
        print("Optimization Finished!")

        y_test1=y_test[np.where(np.argmax(y_test, axis=1) == 1)]
        y_test2=y_test[np.where(np.argmax(y_test, axis=1) == 2)]
        y_test3=y_test[np.where(np.argmax(y_test, axis=1) == 3)]
        y_test4=y_test[np.where(np.argmax(y_test, axis=1) == 4)]
        y_test5=y_test[np.where(np.argmax(y_test, axis=1) == 5)]
        y_test6=y_test[np.where(np.argmax(y_test, axis=1) == 6)]
        y_test7=y_test[np.where(np.argmax(y_test, axis=1) == 7)]
        y_test8=y_test[np.where(np.argmax(y_test, axis=1) == 8)]
        y_test9=y_test[np.where(np.argmax(y_test, axis=1) == 9)]
        y_test0=y_test[np.where(np.argmax(y_test, axis=1) == 0)]

        TP = tf.count_nonzero(predictionEncoded * y)
        TN = tf.count_nonzero((predictionEncoded - 1) * (y - 1))
        FP = tf.count_nonzero(predictionEncoded * (y - 1))
        FN = tf.count_nonzero((predictionEncoded - 1) * y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1_1=f1.eval({x: X_test1, y: y_test1, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_2=f1.eval({x: X_test2, y: y_test2, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_3=f1.eval({x: X_test3, y: y_test3, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_4=f1.eval({x: X_test4, y: y_test4, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_5=f1.eval({x: X_test5, y: y_test5, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_6=f1.eval({x: X_test6, y: y_test6, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_7=f1.eval({x: X_test7, y: y_test7, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_8=f1.eval({x: X_test8, y: y_test8, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_9=f1.eval({x: X_test9, y: y_test9, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})
        f1_0=f1.eval({x: X_test0, y: y_test0, trainingPhase:False, dropoutRateConv:0, dropoutRateFull: 0})

        print('F1 Score(1): '+str(f1_1))
        print('F1 Score(2): '+str(f1_2))
        print('F1 Score(3): '+str(f1_3))
        print('F1 Score(4): '+str(f1_4))
        print('F1 Score(5): '+str(f1_5))
        print('F1 Score(6): '+str(f1_6))
        print('F1 Score(7): '+str(f1_7))
        print('F1 Score(8): '+str(f1_8))
        print('F1 Score(9): '+str(f1_9))
        print('F1 Score(0): '+str(f1_0))

        averageF1 = np.array([f1_1,f1_2,f1_3,f1_4,f1_5,f1_6,f1_7,f1_8,f1_9,f1_0])
        #saver = tf.train.Saver()
        #saver.save(sess, 'classification-model')
        
        return averageF1


# In[56]:


def test(filename):
    image = cv2.imread(filename,1)
    resizedImage = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    resizedImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
    resizedImage = resizedImage.astype('float')
    resizedImage = resizedImage.reshape([-1,32,32,3])
    x = tf.convert_to_tensor(resizedImage)
    tf.GraphKeys.USEFUL = 'useful'
    tf.reset_default_graph()
    new_saver = tf.train.import_meta_graph('classification-model.meta')
    sess = tf.Session() # as sess:
    new_saver.restore(sess, "classification-model")
    
    var_list = tf.get_collection(tf.GraphKeys.USEFUL)
    x_placeholder=var_list[0]
    dropoutRateConv=var_list[1]
    dropoutRateFull=var_list[2]
    trainingPhase=var_list[3]
    predictionTensor=var_list[4]
    result = sess.run(predictionTensor, feed_dict={x_placeholder:resizedImage, dropoutRateConv:0, dropoutRateFull:0, trainingPhase:False})
    return result[0]


# In[57]:

#traintest()
# In[ ]:




