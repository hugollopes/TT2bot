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
def checkFolders(folderPath,folderPathProcessed):
    files = os.listdir(folderPath)
    folderList=[]
    iNumberClasses = len(files)
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
            if (os.path.isfile(full_file_name)):
                shutil.copy(full_file_name, imagefolder)
    return iNumberClasses,folderList
    

def resizeFolderImages(folderPath):
    files = os.listdir(folderPath)
    for image in files:
        imageFile = folderPath + "/" + image
        img = Image.open(imageFile).convert('L')
        img = img.resize((40,40))
        img.save(imageFile)

def Processfile():
    folderPath="/mnt/pythoncode/dataforclassifier/TT2classified"
    folderPathProcessed="/mnt/pythoncode/dataforclassifier/TT2classifiedProcessed"
    print("check and copying data")
    iNumberClasses, folderList = checkFolders(folderPath,folderPathProcessed)
    print("Processing ",iNumberClasses , "classe with paths:",folderList)
    for folder in folderList:
        resizeFolderImages(folder)
    
    
