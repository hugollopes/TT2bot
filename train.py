#import matplotlib.pyplot as plt
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
import tensorflow as tf
import shutil


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

#check for class folders, and copy data to mirror location.
def checkFolders(folderPath,folderPathProcessed,preProcess):
    files = os.listdir(folderPath)
    folderList=[]
    iNumberClasses = len(files)
    iNumberImages= 0
    for folder in files:
        print("image folder:",folder)
        #make copy dirs
        srcFolder = folderPath + "/" +folder
        imagefolder = folderPathProcessed + "/" + folder
        print("imagefolder:",imagefolder)
        ensure_dir(imagefolder)
        folderList.append(imagefolder)
        #copy files
        src_files = os.listdir(srcFolder)
        
        for file_name in src_files:
            full_file_name = os.path.join(srcFolder, file_name)
            if preProcess:
                if (os.path.isfile(full_file_name)):
                    shutil.copy(full_file_name, imagefolder)
            iNumberImages += 1
    return iNumberClasses,folderList,iNumberImages
    
#grays and resizes images
def resizeFolderImages(folderPath, iImgSize,preProcess):
    files = os.listdir(folderPath)
    if preProcess:
        for image in files:
            imageFile = folderPath + "/" + image
            img = Image.open(imageFile).convert('L')
            imageTuple = (iImgSize,iImgSize)
            img = img.resize(imageTuple)
            img.save(imageFile)

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels



def readAndPickle(folders,iImgSize,iNumberImages,pixel_depth):
    dataset,labels = make_arrays(iNumberImages, iImgSize)
    #testfile = folders + "/goldpet/shotnm13.png"
    classFolders = os.listdir(folders)
    labelClass = 0
    imageIndex = 0
    for classfolder in classFolders:
        images = os.listdir(folders + "/" + classfolder)
        for image in images:
            imagefile = folders + "/" + classfolder + "/" + image
            dataset[imageIndex,:,:] = (ndimage.imread(imagefile).astype(float) - 
                pixel_depth / 2) / pixel_depth
            labels[imageIndex] = labelClass
            imageIndex += 1
        labelClass += 1
    dataset, labels = randomize(dataset, labels)
    #print("dataset shape:",dataset.shape,"dataset data",dataset)
    #print("labels shape:",labels.shape,"labels data",labels)
    #separate for train/test/validation
    sizeTrain= int(iNumberImages*0.6)
    sizeTestVal = int(iNumberImages*0.2)
    end_train_index = sizeTrain
    start_valid_index = end_train_index + 1
    end_valid_index = start_valid_index + sizeTestVal
    start_test_index = end_valid_index + 1
    end_test_index =  start_test_index + sizeTestVal
    print("end_train_index: ",end_train_index," end_valid_index: ",end_valid_index," end_test_index: ",end_test_index)

    train_dataset,train_labels = make_arrays(sizeTrain, iImgSize)
    valid_dataset,valid_labels = make_arrays(sizeTestVal, iImgSize)
    test_dataset,test_labels = make_arrays(sizeTestVal, iImgSize)
    train_dataset = dataset[0:end_train_index,:,:]
    train_labels = labels[0:end_train_index]
    valid_dataset = dataset[start_valid_index:end_valid_index,:,:]
    valid_labels = labels[start_valid_index:end_valid_index]
    test_dataset = dataset[start_test_index:end_test_index,:,:]
    test_labels =  labels[start_test_index:end_test_index]
    
    pickle_file = "/mnt/pythoncode/dataforclassifier/" + 'TT2.pickle'
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            }
        print("save:",save)
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


# helper functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def reformat(dataset, labels,image_size,num_labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def trainGraph(image_size,num_labels):
    pickle_file = "/mnt/pythoncode/dataforclassifier/" + 'TT2.pickle'
    batch_size = 64
    print("image_size ",image_size,"num_labels ",num_labels) 


    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = reformat(train_dataset, train_labels,image_size,num_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels,image_size,num_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels,image_size,num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

        
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        
        hidden_layer1_size = 20
        hidden_layer2_size = 10
        hidden_layer3_size = 5
        
        hidden1_weights = weight_variable([image_size * image_size, hidden_layer1_size])
        hidden1_biases= bias_variable([hidden_layer1_size])
        hidden1_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden1_weights) + hidden1_biases)

        hidden2_weights = weight_variable([hidden_layer1_size, hidden_layer2_size ])
        hidden2_biases = bias_variable([hidden_layer2_size])
        hidden2_layer = tf.nn.relu(tf.matmul(hidden1_layer, hidden2_weights) + hidden2_biases)  
        
        hidden3_weights = weight_variable([hidden_layer2_size, hidden_layer3_size ])
        hidden3_biases = bias_variable([hidden_layer3_size])
        hidden3_layer = tf.nn.relu(tf.matmul(hidden2_layer, hidden3_weights) + hidden3_biases)   

        output_weights = weight_variable([hidden_layer3_size, num_labels])
        output_biases = bias_variable([num_labels])
        logits = tf.matmul(hidden3_layer, output_weights) + output_biases

        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        train_prediction = tf.nn.softmax(logits)

        # Setup validation prediction step.        
        valid_hidden1 = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden1_weights) + hidden1_biases)
        valid_hidden2 = tf.nn.relu(tf.matmul(valid_hidden1, hidden2_weights) + hidden2_biases) 
        valid_hidden3 = tf.nn.relu(tf.matmul(valid_hidden2, hidden3_weights) + hidden3_biases)   
        valid_logits = tf.matmul(valid_hidden3, output_weights) + output_biases
        valid_prediction = tf.nn.softmax(valid_logits)

        # And setup the test prediction step.
        test_hidden1 = tf.nn.relu(tf.matmul(tf_test_dataset, hidden1_weights) + hidden1_biases)
        test_hidden2 = tf.nn.relu(tf.matmul(test_hidden1, hidden2_weights) + hidden2_biases)  
        test_hidden3 = tf.nn.relu(tf.matmul(test_hidden2, hidden3_weights) + hidden3_biases)     
        test_logits = tf.matmul(test_hidden3, output_weights) + output_biases
        test_prediction = tf.nn.softmax(test_logits)

    num_steps = 2500

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in xrange(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
              print("Minibatch loss at step", step, ":", l)
              print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
              print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))    

def Processfile():
    preProcess = True
    iImgSize = 40
    pixel_depth = 255.0  # Number of levels per pixel.
    folderPath="/mnt/pythoncode/dataforclassifier/TT2classified"
    folderPathProcessed="/mnt/pythoncode/dataforclassifier/TT2classifiedProcessed"
    print("check and copying data")
    iNumberClasses, folderList,iNumberImages = checkFolders(folderPath,folderPathProcessed,preProcess)
    print("Processing ",iNumberClasses , "classe with paths:",folderList)
    #grays and resizes images
    for folder in folderList:
        resizeFolderImages(folder,iImgSize,preProcess)
    #makes TF ready pickle    
    readAndPickle(folderPathProcessed,iImgSize,iNumberImages,pixel_depth)
    trainGraph(iImgSize,iNumberClasses)

    
    
    
