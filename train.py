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
    print("dataset shape:",dataset.shape,"dataset data",dataset)
    print("labels shape:",labels.shape,"labels data",labels)

def Processfile():
    preProcess = False
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

    
    
    
