from __future__ import print_function
#import numpy as np
import os
import globals as glo
import time
from PIL import Image
from TFFunctions import *
from scipy import ndimage
import sys
"""from IPython.display import display, Image
from scipy import ndimage
from six.moves import cPickle as pickle
import tensorflow as tf
import shutil
from numpy.ma import sqrt"""

#
# goal of this module is to implement generic training/ predictor objects
#

#todo: make debuging in the docker image.


class TrainerPredictor:
    """this class will hold the training and prediction functions of a subset image of TT2bot"""
    base_classification_folder = glo.DATA_FOLDER + "/dataforclassifier"
    images_count = 0

    def __init__(self, name, pred_classes, crop_tuple, size_x, size_y, pixel_depth):
        self.name = name
        self.pred_classes = pred_classes
        self.crop_tuple = crop_tuple
        self.size_x = size_x
        self.size_y = size_y
        self.num_images = 0
        self.pixel_depth = pixel_depth   # Number of levels per pixel.
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


    def process_images(self):
        """this method reads from the classified folder, grays abd resizes, and places in the processed folder"""
        for classification in self.pred_classes:
            image_files = os.listdir(self.classified_folder + "/" + classification)
            for image_file in image_files:
                image = Image.open(self.classified_folder + "/" + classification + '/' + image_file)
                image = image.convert('L')#
                image_tuple = (self.size_x, self.size_y)
                image = image.resize(image_tuple)
                image.save(self.classified_processed + "/" + classification + '/sample'
                           + str(TrainerPredictor.images_count)
                           + '.png')
                TrainerPredictor.images_count += 1
                self.num_images += 1

    def read_and_pickle(self):#folders, iImgSize, iNumberImages, pixel_depth):
        """reads grayed and resized image and creates pickled array of data"""
        dataset, labels = make_arrays(self.num_images,self.size_x,self.size_y)
        label_class = 0
        image_index = 0
        #
        # cycle in for all imager
        #
        for classification in self.pred_classes:
            image_files = os.listdir(self.classified_processed + "/" + classification)
            for image_file in image_files:
                image_file_path = self.classified_processed + "/" + classification + "/" + image_file
                dataset[image_index, :, :] = (ndimage.imread(image_file_path).astype(float) -
                                             pixel_depth / 2) / pixel_depth
        class_folders = os.listdir(folders)

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

        pickle_file = glo.DATA_FOLDER + "/dataforclassifier/" + 'TT2.pickle'
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

boss_trainer = TrainerPredictor("boss_active_predictor", ["boss_active", "boss_inactive", "no_boss"]
                                , (1224, 555, 1248, 648)
                                , 12, 46 , 255.0)
egg_trainer = TrainerPredictor("egg_active_predictor", ["egg_active", "egg_inactive"]
                               , (741, 31, 761, 64)
                               , 10, 16, 255.0)
trainers_predictors_list = []
trainers_predictors_list.append(boss_trainer)
trainers_predictors_list.append(egg_trainer)
for trainer in trainers_predictors_list:
#    trainer.crop_images()
 #   trainer.process_images()



