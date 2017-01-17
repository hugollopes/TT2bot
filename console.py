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


def wait_acknowledge():

    done = "no"
    while done == "no":
        try:
            control_file = open(glo.ACKNOWLEDGE_FILE, "r+")
            control_data = pickle.load(control_file)
            control_file.close()
            done = control_data["Done"]
        except Exception:
            print("error waiting acknowledgment")


def reset_acknowledge():
    try:
        control_data = {"Done": "no"}
        control_file = open(glo.ACKNOWLEDGE_FILE, "wb")
        pickle.dump(control_data, control_file, protocol=2)
        control_file.close()
    except Exception:
        print("some error writing acknowledge file")


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


def attack_command(_argv, _parsed_command):
    if _argv == 2:
        attack_times = int(_parsed_command[1])
        for num in range(0, attack_times):
            insert_command("attack")
    else:
        insert_command("attack")
    reset_acknowledge()
    wait_acknowledge()
    print("command processed")




initialize_control_file()

total_number = get_total_number_files()
print("there are total ", total_number, "files")
sCommands = "(a)ttack,capture,exit,(cg)captureforgold,processfile or ctlr-c':"
print(sCommands)
command = ""
previousCommand = ""
#todo: composed commands like "a 10"
#todo: hold several classifiers.
while str(command) != 'wow':

    command_str = raw_input(">")  # for widows, use input.
    parsed_command = str.split(command_str)
    command = parsed_command[0]
    argv = len(parsed_command)
    if command == "a":
        attack_command(argv, parsed_command)
    if command == "processfile":
        Processfile()
    if command == "capture":
        insert_command("capture")
        reset_acknowledge()
        wait_acknowledge()
    if command == "captureforgold" or command == "cg":
        insert_command("capture")
        reset_acknowledge()
        wait_acknowledge()
        parse_raw_image(total_number, True)
        prediction, total_number = predictpet(total_number)
        if prediction == 0:
            insert_command("getgold")
            print("ready to get gold")

    previousCommand = command
