from __future__ import print_function
import numpy as np
import os
import sys
from IPython.display import display, Image
from scipy import ndimage
from six.moves import cPickle as pickle
from PIL import Image
import tensorflow as tf
import shutil
from numpy.ma import sqrt


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


# check for class folders, and copy data to mirror location.
def check_folder(folderPath, folderPathProcessed, preProcess):
    # first delete all preprocessed
    processedFolderlist = os.listdir(folderPathProcessed)
    for f in processedFolderlist:
        processedFolderlistpath = folderPathProcessed + "/" + f
        processedfolderfiles = os.listdir(processedFolderlistpath)
        for d in processedfolderfiles:
            processedFolderlistfilesPath = folderPathProcessed + "/" + f + "/" + d
            os.remove(processedFolderlistfilesPath)
    print("finish removing files")
    files = os.listdir(folderPath)
    folderList = []
    iNumberClasses = len(files)
    iNumberImages = 0
    for folder in files:
        print("image folder:", folder)
        # make copy dirs
        srcFolder = folderPath + "/" + folder
        imagefolder = folderPathProcessed + "/" + folder
        print("imagefolder:", imagefolder)
        ensure_dir(imagefolder)
        folderList.append(imagefolder)
        # copy files
        src_files = os.listdir(srcFolder)

        for file_name in src_files:
            full_file_name = os.path.join(srcFolder, file_name)
            if preProcess:
                if (os.path.isfile(full_file_name)):
                    shutil.copy(full_file_name, imagefolder)
            iNumberImages += 1
    return iNumberClasses, folderList, iNumberImages


# grays and resizes images
def resizeFolderImages(folderPath, iImgSize, preProcess):
    files = os.listdir(folderPath)
    if preProcess:
        for image in files:
            imageFile = folderPath + "/" + image
            img = Image.open(imageFile).convert('L')
            imageTuple = (iImgSize, iImgSize)
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
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def readAndPickle(folders, iImgSize, iNumberImages, pixel_depth):
    dataset, labels = make_arrays(iNumberImages, iImgSize)
    # testfile = folders + "/goldpet/shotnm13.png"
    classFolders = os.listdir(folders)
    labelClass = 0
    imageIndex = 0
    for classfolder in classFolders:
        images = os.listdir(folders + "/" + classfolder)
        for image in images:
            imagefile = folders + "/" + classfolder + "/" + image
            dataset[imageIndex, :, :] = (ndimage.imread(imagefile).astype(float) -
                                         pixel_depth / 2) / pixel_depth
            labels[imageIndex] = labelClass
            imageIndex += 1
        labelClass += 1
    dataset, labels = randomize(dataset, labels)
    # print("dataset shape:",dataset.shape,"dataset data",dataset)
    # print("labels shape:",labels.shape,"labels data",labels)
    # separate for train/test/validation
    sizeTrain = int(iNumberImages * 0.6)
    sizeTestVal = int(iNumberImages * 0.2)
    end_train_index = sizeTrain
    start_valid_index = end_train_index + 1
    end_valid_index = start_valid_index + sizeTestVal
    start_test_index = end_valid_index + 1
    end_test_index = start_test_index + sizeTestVal
    print("end_train_index: ", end_train_index, " end_valid_index: ", end_valid_index, " end_test_index: ",
          end_test_index)

    train_dataset, train_labels = make_arrays(sizeTrain, iImgSize)
    valid_dataset, valid_labels = make_arrays(sizeTestVal, iImgSize)
    test_dataset, test_labels = make_arrays(sizeTestVal, iImgSize)
    train_dataset = dataset[0:end_train_index, :, :]
    train_labels = labels[0:end_train_index]
    valid_dataset = dataset[start_valid_index:end_valid_index, :, :]
    valid_labels = labels[start_valid_index:end_valid_index]
    test_dataset = dataset[start_test_index:end_test_index, :, :]
    test_labels = labels[start_test_index:end_test_index]

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
        print("save:", save)
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


def reformat(dataset, labels, image_size, num_labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def validateNumHiddenLayers(numHiddenLayers):
    if numHiddenLayers < 1:
        raise ValueError('Number of hidden layers must be >= 1')


def generateHiddenLayerKey(layerNum):
    return 'h' + str(layerNum)


def generateHiddenLayer(layerNum, previousLayer, weights, biases, training, dropoutKeepRate):
    key = generateHiddenLayerKey(layerNum)
    if training:
        hiddenLayer = tf.nn.relu(tf.matmul(previousLayer, weights[key]) + biases[key])
        hiddenLayer = tf.nn.dropout(hiddenLayer, dropoutKeepRate)
        return hiddenLayer
    else:
        hiddenLayer = tf.nn.relu(tf.matmul(previousLayer, weights[key]) + biases[key])
        return hiddenLayer


def multilayerNetwork(inputs, weights, biases, numHiddenLayers, training, dropoutKeepRate):
    validateNumHiddenLayers(numHiddenLayers)

    hiddenLayer = generateHiddenLayer(1, inputs, weights, biases, training, dropoutKeepRate)

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            hiddenLayer = generateHiddenLayer(layerNum, hiddenLayer, weights, biases, training, dropoutKeepRate)

    return tf.matmul(hiddenLayer, weights['out']) + biases['out']


def reformat(dataset, labels, _imageSize, _numLabels):
    dataset = dataset.reshape((-1, _imageSize * _imageSize)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(_numLabels) == labels[:, None]).astype(np.float32)
    return dataset, labels


# source:  http://arxiv.org/pdf/1502.01852v1.pdf
def calculateOptimalWeightStdDev(numPreviousLayerParams):
    return sqrt(2.0 / numPreviousLayerParams)


def generateWeights(hiddenLayers, numInputs, numLabels):
    numHiddenLayers = hiddenLayers.__len__()
    validateNumHiddenLayers(numHiddenLayers)
    weights = {}

    numHiddenFeatures = hiddenLayers[0]
    stddev = calculateOptimalWeightStdDev(numInputs)
    weights[generateHiddenLayerKey(1)] = tf.Variable(tf.truncated_normal([numInputs, numHiddenFeatures], 0, stddev))

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            previousNumHiddenFeatures = numHiddenFeatures
            numHiddenFeatures = hiddenLayers[layerNum - 1]
            stddev = calculateOptimalWeightStdDev(previousNumHiddenFeatures)
            weights[generateHiddenLayerKey(layerNum)] = tf.Variable(
                tf.truncated_normal([previousNumHiddenFeatures, numHiddenFeatures], 0, stddev))

    stddev = calculateOptimalWeightStdDev(numHiddenFeatures)
    weights['out'] = tf.Variable(tf.truncated_normal([numHiddenFeatures, numLabels], 0, stddev))
    return weights


def generateBiases(hiddenLayers, numLabels):
    numHiddenLayers = hiddenLayers.__len__()
    validateNumHiddenLayers(numHiddenLayers)
    biases = {}

    numHiddenFeatures = hiddenLayers[0]
    biases[generateHiddenLayerKey(1)] = tf.Variable(tf.zeros([numHiddenFeatures]))

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            numHiddenFeatures = hiddenLayers[layerNum - 1]
            biases[generateHiddenLayerKey(layerNum)] = tf.Variable(tf.zeros([numHiddenFeatures]))

    biases['out'] = tf.Variable(tf.zeros([numLabels]))
    return biases


def generateRegularizers(weights, biases, numHiddenLayers):
    validateNumHiddenLayers(numHiddenLayers)
    regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['h1'])

    for layerNum in xrange(numHiddenLayers + 1):
        if layerNum > 1:
            regularizers = regularizers + tf.nn.l2_loss(weights['h' + str(layerNum)]) + tf.nn.l2_loss(
                biases['h' + str(layerNum)])

    regularizers = regularizers + tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out'])
    return regularizers


def generateLossCalc(weights, biases, numHiddenLayers, trainingNetwork, trainingLabels, regularizationRate):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(trainingNetwork, trainingLabels))
    regularizers = generateRegularizers(weights, biases, numHiddenLayers)
    loss += regularizationRate * regularizers
    return loss


def trainGraph(image_size, num_labels):
    pickle_file = "/mnt/pythoncode/dataforclassifier/" + 'TT2.pickle'
    model_file = "/mnt/pythoncode/dataforclassifier/" + 'petdetectionmodel.ckpt'
    batch_size = 128
    print("image_size ", image_size, "num_labels ", num_labels)

    _imageSize = image_size
    _numLabels = num_labels
    _trainSubset = 5000
    _batchSize = 128
    _hiddenLayers = [200, 30]
    _numInputs = _imageSize * _imageSize
    _startLearningRate = 0.5
    _learningDecayRate = 0.98
    _decaySteps = 1000
    _numSteps = 2500
    _regularizationRate = 0.00001
    _dropoutKeepRate = 1

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

    train_dataset, train_labels = reformat(train_dataset, train_labels, _imageSize, _numLabels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, _imageSize, _numLabels)
    test_dataset, test_labels = reformat(test_dataset, test_labels, _imageSize, num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(_batchSize, _numInputs))
        tf_train_labels = tf.placeholder(tf.float32, shape=(_batchSize, _numLabels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        numHiddenLayers = _hiddenLayers.__len__()
        weights = generateWeights(_hiddenLayers, _numInputs, _numLabels)
        print(weights)
        biases = generateBiases(_hiddenLayers, _numLabels)
        trainingNetwork = multilayerNetwork(tf_train_dataset, weights, biases, numHiddenLayers, True, _dropoutKeepRate)
        loss = generateLossCalc(weights, biases, numHiddenLayers, trainingNetwork, tf_train_labels, _regularizationRate)
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(_startLearningRate, global_step, _decaySteps, _learningDecayRate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        train_prediction = tf.nn.softmax(
            multilayerNetwork(tf_train_dataset, weights, biases, numHiddenLayers, False, _dropoutKeepRate))
        valid_prediction = tf.nn.softmax(
            multilayerNetwork(tf_valid_dataset, weights, biases, numHiddenLayers, False, _dropoutKeepRate))
        test_prediction = tf.nn.softmax(
            multilayerNetwork(tf_test_dataset, weights, biases, numHiddenLayers, False, _dropoutKeepRate))

        oSaver = tf.train.Saver()
        all_vars = tf.trainable_variables()
        for v in all_vars:
            print(v.name)
            print(v.value())

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in xrange(_numSteps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (np.random.randint(1, _trainSubset) * _batchSize) % (train_labels.shape[0] - _batchSize)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + _batchSize), :]
            batch_labels = train_labels[offset:(offset + _batchSize), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 500 == 0):
                print("Minibatch loss at step", step, ":", l)
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        oSaver.save(session, model_file)  # filename ends with .ckpt


def Processfile():
    preProcess = True  # read again the raw files?
    iImgSize = 40
    pixel_depth = 255.0  # Number of levels per pixel.
    folderPath = "/mnt/pythoncode/dataforclassifier/TT2classified"
    folderPathProcessed = "/mnt/pythoncode/dataforclassifier/TT2classifiedProcessed"
    print("check and copying data")
    iNumberClasses, folderList, iNumberImages = check_folder(folderPath, folderPathProcessed, preProcess)
    print("Processing ", iNumberClasses, "classe with paths:", folderList)
    # grays and resizes images
    for folder in folderList:
        resizeFolderImages(folder, iImgSize, preProcess)
    # makes TF ready pickle
    readAndPickle(folderPathProcessed, iImgSize, iNumberImages, pixel_depth)
    trainGraph(iImgSize, iNumberClasses)


# getsingleimage
def getSingleImageTensor(img_size, pixel_depth):
    imagefile = "/mnt/pythoncode/detect.png"
    imagefileResized = "/mnt/pythoncode/detectResized.png"
    img = Image.open(imagefile).convert('L')
    imageTuple = (img_size, img_size)
    img = img.resize(imageTuple)
    img.save(imagefileResized)

    image = np.ndarray((1, img_size, img_size), dtype=np.float32)

    image[0, :, :] = (ndimage.imread(imagefileResized).astype(float) - pixel_depth / 2) / pixel_depth
    image = image.reshape((-1, img_size * img_size)).astype(np.float32)
    return image


def savePrediction(filecounts, prediction):
    print("saving file number", filecounts)
    if prediction == 0:
        path = "goldpet"
    if prediction == 1:
        path = "nopet"
    if prediction == 2:
        path = "normalpet"
    if prediction == 3:
        path = "partial pet"
    imagefile = "/mnt/pythoncode/detect.png"
    targetfile = "/mnt/pythoncode/dataforclassifier/TT2predictionsamples/" + path + "/sample" + str(filecounts) + ".png"
    # print("targetfile",targetfile)
    shutil.copy(imagefile, targetfile)


import operator


def getmax(l):
    max_idx, max_val = max(enumerate(l), key=operator.itemgetter(1))
    return max_idx, max_val


def predictpet(filecounts):
    _hiddenLayers = [200, 30]
    _imageSize = 40
    pixel_depth = 255.0
    model_file = "/mnt/pythoncode/dataforclassifier/" + 'petdetectionmodel.ckpt.meta'
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(model_file)
    new_saver.restore(sess, tf.train.latest_checkpoint('/mnt/pythoncode/dataforclassifier/'))
    all_vars = tf.trainable_variables()
    tf_image_dataset = tf.placeholder(tf.float32, shape=(1, _imageSize * _imageSize))
    h1 = all_vars[0]
    h2 = all_vars[1]
    out = all_vars[2]
    b1 = all_vars[3]
    b2 = all_vars[4]
    bOut = all_vars[5]

    hiddenLayer1 = tf.nn.relu(tf.matmul(tf_image_dataset, h1) + b1)
    hiddenLayer2 = tf.nn.relu(tf.matmul(hiddenLayer1, h2) + b2)
    finalLayer = tf.nn.softmax(tf.nn.relu(tf.matmul(hiddenLayer2, out) + bOut))

    predict_dataset = getSingleImageTensor(_imageSize, pixel_depth)
    predictions = sess.run(finalLayer, feed_dict={tf_image_dataset: predict_dataset})
    prediction, max_val = getmax(predictions[0])
    savePrediction(filecounts, prediction)
    print("prediction:", prediction)
    filecounts += 1
    return prediction, filecounts
