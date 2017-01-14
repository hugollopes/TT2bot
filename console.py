import pickle
import sys
import os
sys.path.append(os.path.abspath("/mnt/pythoncode"))
from train import Processfile,predictpet


dControlData = {}

#for windows:
#strControlFile = "C:\\Users\\TGDLOHU1\\pythoncode\\controlaction.txt"
#for within docker image
strControlFile = "/mnt/pythoncode/controlaction.txt"

dControlData = { "Action": "nothing", "AttackNumber": 300 }



#controlFile = open( strControlFile, "rb" )
#dControlData = pickle.load( controlFile )
#print (dControlData['Action'])
#controlFile.close()

sCommands ="(a)ttack,capture,capturepet,(d)etectpet,exit,processfile or ctlr-c':"
print(sCommands)
command = ""
previousCommand = ""
while str(command) != 'wow':
    command = raw_input(">")#for widows, use input.
    if command == "a":
        command = "attack"
    if command == "processfile":
        Processfile()
        command = previousCommand
    if command == "detectpet" or command == "d":
        predictpet()
        command = previousCommand
    dControlData['Action'] = str(command)
    controlFile = open( strControlFile, "wb" )
    pickle.dump( dControlData, controlFile ,protocol=2)
    controlFile.close()
    previousCommand = command

    




