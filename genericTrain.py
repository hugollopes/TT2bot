from __future__ import print_function
#import numpy as np
import os
import globals as glo
import time
import tensorflow as tf
from PIL import Image
from TFFunctions import *
from scipy import ndimage
import shutil



#
# goal of this module is to implement generic training/ predictor objects
#

#todo: make debuging in the docker image.


class TrainerPredictor:
    """this class will hold the training and prediction functions of a subset image of TT2bot"""
    base_classification_folder = glo.DATA_FOLDER + "/dataforclassifier"
    images_count = 0

    def __init__(self, name, pred_classes, crop_tuple, size_x, size_y, pixel_depth, hidden_layers, **kwargs):
        self.name = name
        self.pred_classes = pred_classes
        self.num_classes = len(self.pred_classes)
        self.crop_tuple = crop_tuple
        self.size_x = size_x
        self.size_y = size_y
        self.num_images = 0
        self.pixel_depth = pixel_depth   # Number of levels per pixel.
        #
        # learning parameters
        #
        self.batch_size = 128
        self.hidden_layers = hidden_layers
        self.num_features = self.size_x * self.size_y
        self.start_learning_rate = 0.5
        self.learning_decay_rate = 0.98
        self.decay_Steps = 1000
        self.num_Steps = 2500
        self.regularization_rate = 0.00001
        self.dropout_keep_rate = 1
        #
        # create folders
        #
        self.base_folder = TrainerPredictor.base_classification_folder + "/" + name
        self.classified_folder = self.base_folder + "/classified"
        self.classified_processed = self.base_folder + "/classified_processed"
        self.unclassified_folder = self.base_folder + "/unclassified"
        self.prediction_samples_folder = self.base_folder + "/prediction_samples"

        self.classification_folders = []
        self.classification_folders.append(self.base_folder)
        self.classification_folders.append(self.classified_folder)
        self.classification_folders.append(self.classified_processed)
        self.classification_folders.append(self.unclassified_folder)
        self.classification_folders.append(self.prediction_samples_folder)

        for classification in self.pred_classes:
            self.classification_folders.append(self.classified_folder + "/" + classification)
            self.classification_folders.append(self.classified_processed + "/" + classification)
            self.classification_folders.append(self.prediction_samples_folder + "/" + classification)

        for folder in self.classification_folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print("creating directory", folder)
        #
        # important files
        #
        self.pickle_processed_images_file = self.base_folder + "/processed_images.pickle"
        self.pickle_tf_model_file = self.base_folder + "/tf_model.ckpt"
        self.tf_meta_model_file = self.pickle_tf_model_file + ".meta"
        self.prediction_image_file = self.base_folder + '/' + self.name + 'prediction_crop.png'
        self.prediction_processed_image_file = self.base_folder + '/' + self.name + 'prediction_processed_crop.png'

    def crop_images(self):
        """ this function reads raw files, crops images, and places them in the unclassified folder"""
        print("cropping", self.name, ":", str(self.crop_tuple))
        image_files = os.listdir(glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER)
        for image_file in image_files:
            image = Image.open(glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER + '/' + image_file)
            crop = image.crop(self.crop_tuple)
            crop.save(self.unclassified_folder + '/' + self.name + 'cropped' + str(TrainerPredictor.images_count)
                      + '.png')
            TrainerPredictor.images_count += 1

    def predict_crop(self,image):
        crop = image.crop(self.crop_tuple)
        crop.save(self.prediction_image_file) # save original crop
        crop = crop.convert('L')
        crop = crop.resize((self.size_x, self.size_y))
        crop.save(self.prediction_processed_image_file) # save resized and grayed crop

    def process_images(self):
        """this method deletes the processed folder,
        reads from the classified folder, grays abd resizes, and places in the processed folder"""
        #
        # first delete all processed
        #
        for classification in self.pred_classes:
            image_files = os.listdir(self.classified_processed + "/" + classification)
            for image_file in image_files:
                os.remove(self.classified_processed + "/" + classification + '/' + image_file)
        print("finish removing files")
        #
        # gray and resize and save in the processed.
        #
        for classification in self.pred_classes:
            image_files = os.listdir(self.classified_folder + "/" + classification)
            for image_file in image_files:
                image = Image.open(self.classified_folder + "/" + classification + '/' + image_file)
                image = image.convert('L')
                image_tuple = (self.size_x, self.size_y)
                image = image.resize(image_tuple)
                image.save(self.classified_processed + "/" + classification + '/sample'
                           + str(TrainerPredictor.images_count)
                           + '.png')
                TrainerPredictor.images_count += 1
                self.num_images += 1

    def read_and_pickle(self):
        """reads grayed and resized image and creates pickled array of data"""
        dataset, labels = make_arrays(self.num_images, self.size_y, self.size_x)
        label_class = 0
        image_index = 0
        #
        # cycle in for all processed classified images
        #
        for classification in self.pred_classes:
            image_files = os.listdir(self.classified_processed + "/" + classification)
            for image_file in image_files:
                image_file_path = self.classified_processed + "/" + classification + "/" + image_file
                dataset[image_index, :, :] = (ndimage.imread(image_file_path).astype(float)
                                              - self.pixel_depth / 2) / self.pixel_depth
                labels[image_index] = label_class
                image_index += 1
            label_class += 1

        dataset, labels = randomize(dataset, labels)
        # print("dataset shape:",dataset.shape,"dataset data",dataset)
        # print("labels shape:",labels.shape,"labels data",labels)
        # separate for train/test/validation
        size_train = int(self.num_images * 0.6)
        size_test_val = int(self.num_images * 0.2)
        end_train_index = size_train
        start_valid_index = end_train_index + 1
        end_valid_index = start_valid_index + size_test_val
        start_test_index = end_valid_index + 1
        end_test_index = start_test_index + size_test_val

        #print("end_train_index: ", end_train_index, " end_valid_index: ", end_valid_index, " end_test_index: ",
        #      end_test_index)

        train_dataset = dataset[0:end_train_index, :, :]
        train_labels = labels[0:end_train_index]
        valid_dataset = dataset[start_valid_index:end_valid_index, :, :]
        valid_labels = labels[start_valid_index:end_valid_index]
        test_dataset = dataset[start_test_index:end_test_index, :, :]
        test_labels = labels[start_test_index:end_test_index]

        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }

        save_pickle(self.pickle_processed_images_file, save)

    def train_graph(self):
        print("image_size ", self.size_x ," ", self.size_y, "num_labels ", self.num_classes)

        _trainSubset = self.num_images # for now allways the same.

        with open(self.pickle_processed_images_file, 'rb') as f:
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

        train_dataset, train_labels = reformat(train_dataset, train_labels, self.size_x, self.size_y, self.num_classes)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, self.size_x, self.size_y, self.num_classes)
        test_dataset, test_labels = reformat(test_dataset, test_labels, self.size_x, self.size_y, self.num_classes)
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        #
        # fix batch size for very little data.
        #
        if len(train_dataset) < self.batch_size:
            self.batch_size = 24
            print("batch size changed", self.batch_size)

        tf.reset_default_graph()

        graph = tf.Graph()
        with graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed
            # at run time with a training minibatch.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_features))
            tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_classes))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            num_hidden_layers = self.hidden_layers.__len__()
            weights = generate_weights(self.hidden_layers, self.num_features, self.num_classes)
            #print(weights)
            biases = generate_biases(self.hidden_layers, self.num_classes)
            training_network = multilayer_network(tf_train_dataset, weights, biases, num_hidden_layers
                                                  , True, self.dropout_keep_rate)
            loss = generate_loss_calc(weights, biases, num_hidden_layers, training_network
                                      , tf_train_labels, self.regularization_rate)
            global_step = tf.Variable(0)  # count the number of steps taken.
            learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step
                                                       , self.decay_Steps, self.start_learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            train_prediction = tf.nn.softmax(
                multilayer_network(tf_train_dataset, weights, biases, num_hidden_layers, False, self.dropout_keep_rate))
            valid_prediction = tf.nn.softmax(
                multilayer_network(tf_valid_dataset, weights, biases, num_hidden_layers, False, self.dropout_keep_rate))
            test_prediction = tf.nn.softmax(
                multilayer_network(tf_test_dataset, weights, biases, num_hidden_layers, False, self.dropout_keep_rate))

            oSaver = tf.train.Saver()
            all_vars = tf.trainable_variables()
            for v in all_vars:
                print(v.name)
                print(v.value())

        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in xrange(self.num_Steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (np.random.randint(1, _trainSubset) * self.batch_size) % (train_labels.shape[0]
                                                                                   - self.batch_size)
                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + self.batch_size), :]
                batch_labels = train_labels[offset:(offset + self.batch_size), :]
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
            oSaver.save(session, self.pickle_tf_model_file)  # filename ends with .ckpt
            session.close()

    def get_single_image_tensor(self):#img_size, pixel_depth):
        image_tensor = np.ndarray((1, self.size_y, self.size_x), dtype=np.float32)
        image_tensor[0, :, :] = (ndimage.imread(self.prediction_processed_image_file).astype(float)
                                 - self.pixel_depth / 2) / self.pixel_depth
        image_tensor = image_tensor.reshape((-1, self.size_x * self.size_y)).astype(np.float32)
        return image_tensor

    def save_prediction(self, prediction):
        target_file = self.prediction_samples_folder + "/" + str(self.pred_classes[int(prediction)]) \
                      + "/sample" + time.strftime("%Y%m%d-%H%M%S-%f") + ".png"
        # print("targetfile",targetfile)
        shutil.copy(self.prediction_image_file, target_file)

    def predict_parsed(self):#filecounts):
        print("predicting ",self.name)
        tf.reset_default_graph()
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(self.tf_meta_model_file)
        new_saver.restore(sess, tf.train.latest_checkpoint(self.base_folder))
        all_vars = tf.trainable_variables()
        tf_image_dataset = tf.placeholder(tf.float32, shape=(1, self.size_x * self.size_y))
        h1 = all_vars[0]
        h2 = all_vars[1]
        out = all_vars[2]
        b1 = all_vars[3]
        b2 = all_vars[4]
        bOut = all_vars[5]

        hiddenLayer1 = tf.nn.relu(tf.matmul(tf_image_dataset, h1) + b1)
        hiddenLayer2 = tf.nn.relu(tf.matmul(hiddenLayer1, h2) + b2)
        finalLayer = tf.nn.softmax(tf.nn.relu(tf.matmul(hiddenLayer2, out) + bOut))

        predict_dataset = self.get_single_image_tensor()
        predictions = sess.run(finalLayer, feed_dict={tf_image_dataset: predict_dataset})
        sess.close()
        prediction, max_val = get_max(predictions[0])
        self.save_prediction(prediction)
        print(self.name, " prediction:", prediction)
        return prediction


class TT2Predictor:
    """holds the several trainer predictor instances and common operations """

    def __init__(self, **kwargs):
        self.trainers_predictors_list = []
        boss_trainer = TrainerPredictor("boss_active_predictor", ["boss_active", "boss_inactive", "no_boss"]
                                        , (1224, 555, 1248, 648)
                                        , 12, 46, 255.0
                                        , [200, 30])
        egg_trainer = TrainerPredictor("egg_active_predictor", ["egg_active", "egg_inactive"]
                                       , (741, 31, 761, 64)
                                       , 10, 16, 255.0
                                       , [200, 30])
        gold_pet_trainer = TrainerPredictor("gold_pet_predictor", ["goldpet", "nopet", "normalpet", "partial pet"]
                                            , (624, 364, 734, 474)
                                            , 40, 40, 255.0
                                            , [200, 30])
        self.trainers_predictors_list.append(boss_trainer)
        self.trainers_predictors_list.append(egg_trainer)
        self.trainers_predictors_list.append(gold_pet_trainer)
        for trainer in self.trainers_predictors_list:
            pass
            #trainer.crop_images()
            #trainer.process_images()
            #trainer.read_and_pickle()
            #trainer.train_graph()
        saved_classes_file = glo.DATA_FOLDER + "/dataforclassifier/TrainerPredictor_list.pickle"
        save_pickle(saved_classes_file, self.trainers_predictors_list)

    def parse_raw_image(self):
        start = time.time()
        with open(glo.RAW_FULL_FILE, 'rb') as f:
            image = Image.frombytes('RGBA', (1280, 720), f.read())
        for class_predictor in self.trainers_predictors_list:
            class_predictor.predict_crop(image)
        image.save(glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER + "/fullcapture"
                   + time.strftime("%Y%m%d-%H%M%S-%f") + ".png")  # save original capture copy
        print("parse and crop time: ", time.time() - start)

    def predict_parsed_all(self):
        pred_dict = {}
        for class_predictor in self.trainers_predictors_list:
            pred_dict[class_predictor.name] = class_predictor.predict_parsed()
        return pred_dict

    def predict(self):
        self.parse_raw_image()
        return self.predict_parsed_all()

    def check_predict(self, pred_dict, predictor, classification):
        for class_predictor in self.trainers_predictors_list:
            if class_predictor.name == predictor:
                return int(pred_dict[predictor]) == class_predictor.pred_classes.index(classification)




"""
predictor = TT2Predictor()
predictor.parse_raw_image()
predictor.predict_parsed_all()
"""

