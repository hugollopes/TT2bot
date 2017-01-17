import os
import pickle
import sys
import time
import globals as glo
from PIL import Image
from train import Processfile, predictpet




# this function returns the total increment of classified and unclassified pet pictures
def get_total_number_files():
    count_total_number = 0
    file_paths = [glo.PET_PREDICTION_SAMPLES_FOLDER + "/goldpet",
                  glo.PET_PREDICTION_SAMPLES_FOLDER + "/nopet",
                  glo.PET_PREDICTION_SAMPLES_FOLDER + "/normalpet",
                  glo.PET_PREDICTION_SAMPLES_FOLDER + "/partial pet",
                  glo.PET_CLASSIFIED_DATA_FOLDER + "/goldpet",
                  glo.PET_CLASSIFIED_DATA_FOLDER + "/nopet",
                  glo.PET_CLASSIFIED_DATA_FOLDER + "/normalpet",
                  glo.PET_CLASSIFIED_DATA_FOLDER + "/partial pet",
                  glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER]
    for path in file_paths:
        src_files = os.listdir(path)
        for _ in src_files:
            count_total_number += 1
    return count_total_number

#create more code modules to hold code.
def parse_raw_image(total_number, saveoriginal):
    start = time.time()
    # test raw reading
    iImgSize = 40
    with open(glo.RAW_FULL_FILE, 'rb') as f:
        im = Image.frombytes('RGBA', (1280, 720), f.read())
    pet_crop = im.crop((624, 364, 624 + 110, 364 + 110))
    pet_crop.save(glo.DATA_FOLDER + '/detect.png')
    pet_crop = pet_crop.convert('L')
    imageTuple = (iImgSize, iImgSize)
    pet_crop = pet_crop.resize(imageTuple)
    pet_crop.save(glo.DATA_FOLDER + '/detectResized.png')
    if saveoriginal:
        im.save(glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER + "/fullcapture" + str(total_number) + " .png")
    end = time.time()
    print("captureImage time: ", end - start)


def insert_command(_command):
    try:
        control_file = open(glo.CONTROL_FILE, "rb")
        control_data = pickle.load(control_file)
        control_file.close()

        command_list = control_data["Action"]
        command_list.insert(0, _command)
        control_data["Action"] = command_list
        control_file = open(glo.CONTROL_FILE, "wb")
        pickle.dump(control_data, control_file, protocol=2)
        control_file.close()
    except Exception:
        print("some error inserting command. action=nothing")


def initialize_control_file():
    action_list = ["nothing"]
    control_data = {"Action": action_list, "AttackNumber": 300}
    control_file = open(glo.CONTROL_FILE, "wb")
    pickle.dump(control_data, control_file, protocol=2)
    control_file.close()
    print("file initialized")

initialize_control_file()

total_number = get_total_number_files()
print("there are total ", total_number, "files")
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
        insert_command("attack")
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
            controlFile = open(glo.CONTROL_FILE, "wb")
            pickle.dump(dControlData, controlFile, protocol=2)
            controlFile.close()

            time.sleep(2)  # wait for android processing to optaing the raw image
            parse_raw_image(total_number, True)
            time.sleep(0.3)
            prediction, total_number = predictpet(total_number)
            if prediction == 0:
                dControlData['Action'] = str("getgold")
                controlFile = open(glo.CONTROL_FILE, "wb")
                pickle.dump(dControlData, controlFile, protocol=2)
                controlFile.close()
                print("ready to get gold")
                time.sleep(1)  # wait for not overwritten

    #dControlData['Action'] = str(command)
    #controlFile = open(glo.CONTROL_FILE, "wb")
    #pickle.dump(dControlData, controlFile, protocol=2)
    #controlFile.close()
    if command == "detectpet":
        time.sleep(5)
        prediction, total_number = predictpet(total_number)
        if prediction == 0:
            dControlData['Action'] = str("getgold")
            controlFile = open(glo.CONTROL_FILE, "wb")
            pickle.dump(dControlData, controlFile, protocol=2)
            controlFile.close()
            print("ready to get gold")

    previousCommand = command
