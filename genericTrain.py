from __future__ import print_function
#import numpy as np
import os
import globals as glo
import sys
"""from IPython.display import display, Image
from scipy import ndimage
from six.moves import cPickle as pickle
from PIL import Image
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

    def __init__(self, name, pred_classes):
        self.name = name
        self.pred_classes = pred_classes
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

        def preprocess_images:
            """ this function reads raw files, crops images, and places them in the unclassified folder"""
            #todo: complete...



boss_trainer = TrainerPredictor("boss_active_predictor", ["boss_active", "boss_inactive", "no_boss"],)
egg_trainer = TrainerPredictor("egg_active_predictor", ["egg_active", "egg_inactive"])
trainers_predictors_list = []
trainers_predictors_list.append(1)
trainers_predictors_list.append(boss_trainer)
trainers_predictors_list.append(egg_trainer)
print(trainers_predictors_list[1].name)


