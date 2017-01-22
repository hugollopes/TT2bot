from __future__ import print_function
import os
import pickle
import sys
import time
import globals as glo
from PIL import Image
#from train import Processfile, predictpet
from genericTrain import TT2Predictor
from TT2actions import *

"""
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
"""



def initialize_control_file():
    action_list = [{"Type": "nothing"}]
    control_data = {"ActionList": action_list}
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

predictor = TT2Predictor()

initialize_control_file()

#total_number = get_total_number_files()
#print("there are total ", total_number, "files")




#todo: dynamic prediction hidden layer configuration
#todo:generic drag command
#todo: recognize level number
#todo:  recognize active tabs
#todo: recognize heroes position
#todo: reconize heroes data
#todo: increment hero
#todo: store list of actions
#todo: ml on actions


sCommands = "(a)ttack,capture,exit,(cg)captureforgold,processfile,(r)ecognize or ctlr-c':"
print(sCommands)
command = ""
previousCommand = ""
while str(command) != 'wow':

    command_str = raw_input(">")  # for widows, use input.
    parsed_command = str.split(command_str)
    command = parsed_command[0]
    argv = len(parsed_command)
    if command == "a":
        attack_command(argv, parsed_command)
    if command == "hit":
        insert_command("hit", X=parsed_command[1], Y=parsed_command[2])
        reset_acknowledge()
        wait_acknowledge()
    #if command == "processfile":
    #    Processfile()
    if command == "capture":
        insert_command("capture")
        reset_acknowledge()
        wait_acknowledge()
    if command == "recognize" or command == "r":
        reset_acknowledge()
        insert_command("capture")
        wait_acknowledge()
        predictor.parse_raw_image()
        pred_dict = predictor.predict_parsed_all()
        if int(pred_dict['egg_active_predictor']) == 0:
            print("capturing egg")
            insert_command("hit", X=glo.HIT_DICT["egg"][0], Y=glo.HIT_DICT["egg"][1])
            time.sleep(0.5)
            insert_command("hit", X=glo.HIT_DICT["Shinning_egg"][0], Y=glo.HIT_DICT["Shinning_egg"][1])
            time.sleep(0.5)
            insert_command("hit", X=glo.HIT_DICT["Shinning_egg"][0], Y=glo.HIT_DICT["Shinning_egg"][1])
            time.sleep(1)
            insert_command("hit", X=glo.HIT_DICT["Shinning_egg"][0], Y=glo.HIT_DICT["Shinning_egg"][1]) #one more hit to clear
            time.sleep(1)
            insert_command("hit", X=glo.HIT_DICT["Shinning_egg"][0], Y=glo.HIT_DICT["Shinning_egg"][1])  # one more hit to clear
    if command == "captureforgold" or command == "cg":
        capture_gold_forever(predictor)
    previousCommand = command
