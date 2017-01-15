import os
import pickle
import sys
import time

from PIL import Image
from globals import *
from train import Processfile, predictpet

_data_folder, _share_folder , _code_folder, _pet_prediction_samples_folder, _pet_classified_data_folder, _unclassified_global_captures_folder , _control_file , _raw_full_file = set_globals()
sys.path.append(os.path.abspath(_code_folder))


# this function returns the total increment of classified and unclassified pet pictures
def get_total_number_files():
    total_number = 0
    file_paths = [_pet_prediction_samples_folder + "/goldpet",
                  _pet_prediction_samples_folder + "/nopet",
                  _pet_prediction_samples_folder + "/normalpet",
                  _pet_prediction_samples_folder + "/partial pet",
                  _pet_classified_data_folder + "/goldpet",
                  _pet_classified_data_folder + "/nopet",
                  _pet_classified_data_folder + "/normalpet",
                  _pet_classified_data_folder + "/partial pet",
                  _unclassified_global_captures_folder]
    for path in file_paths:
        src_files = os.listdir(path)
        for _ in src_files:
            total_number += 1
    return total_number

#create more code modules to hold code.
def parse_raw_image(total_number, saveoriginal):
    start = time.time()
    # test raw reading
    iImgSize = 40
    with open(_raw_full_file, 'rb') as f:
        im = Image.frombytes('RGBA', (1280, 720), f.read())
    pet_crop = im.crop((624, 364, 624 + 110, 364 + 110))
    pet_crop.save(_data_folder + '/detect.png')
    pet_crop = pet_crop.convert('L')
    imageTuple = (iImgSize, iImgSize)
    pet_crop = pet_crop.resize(imageTuple)
    pet_crop.save(_data_folder + '/detectResized.png')
    if saveoriginal:
        im.save(_unclassified_global_captures_folder + "/fullcapture" + str(total_number) + " .png")
    end = time.time()
    print("captureImage time: ", end - start)


total_number = get_total_number_files()
print("there are total ", total_number, "files")
dControlData = {}

dControlData = {"Action": "nothing", "AttackNumber": 300}

sCommands = "(a)ttack,capture,capturepet,(d)etectpet,exit,(cg)captureforgold,processfile or ctlr-c':"
print(sCommands)
command = ""
previousCommand = ""
#todo: decent action pipe between processes.
#todo: use proper primitives to access files
#todo: proper global variables handling
#todo: hold several classifiers.
while str(command) != 'wow':

    if previousCommand == "detectpet":
        time.sleep(5)
        Command = "detectpet"
    else:
        command = raw_input(">")  # for widows, use input.
    if command == "a":
        command = "attack"
    if command == "processfile":
        Processfile()
        command = previousCommand
    if command == "detectpet" or command == "d":
        command = "detectpet"
    if command == "getgold" or command == "g":
        pass
    if command == "captureforgold" or command == "cg":
        forever = True
        while forever:
            dControlData['Action'] = str("capture")
            controlFile = open(_control_file, "wb")
            pickle.dump(dControlData, controlFile, protocol=2)
            controlFile.close()

            time.sleep(2)  # wait for android processing to optaing the raw image
            parse_raw_image(total_number, True)
            time.sleep(0.3)
            prediction, total_number = predictpet(total_number)
            if prediction == 0:
                dControlData['Action'] = str("getgold")
                controlFile = open(_control_file, "wb")
                pickle.dump(dControlData, controlFile, protocol=2)
                controlFile.close()
                print("ready to get gold")
                time.sleep(1)  # wait for not overwritten

    dControlData['Action'] = str(command)
    controlFile = open(_control_file, "wb")
    pickle.dump(dControlData, controlFile, protocol=2)
    controlFile.close()
    if command == "detectpet":
        time.sleep(5)
        prediction, total_number = predictpet(total_number)
        if prediction == 0:
            dControlData['Action'] = str("getgold")
            controlFile = open(_control_file, "wb")
            pickle.dump(dControlData, controlFile, protocol=2)
            controlFile.close()
            print("ready to get gold")

    previousCommand = command
