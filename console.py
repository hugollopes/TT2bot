import pickle
import sys
import os
import time
sys.path.append(os.path.abspath("/mnt/pythoncode"))
from train import Processfile,predictpet

def getFileCounts():
    filecounts = [0,0,0,0]
    i=0
    for i in range(0,len(filecounts)):
        if i==0:
            path= "goldpet"
        if i==1:
            path ="nopet"
        if i==2:
            path ="normalpet"
        if i==3:
            path ="partialpet"
        x=0
        while os.path.exists("/mnt/pythoncode/dataforclassifier/TT2predictionsamples/" + path + "/sample%s.png" % x):
            x +=1
        filecounts[i] = x
        i +=1    
    #print(filecounts)
    return filecounts    


filecounts = getFileCounts()
dControlData = {}

#for windows: #strControlFile = "C:\\Users\\TGDLOHU1\\pythoncode\\controlaction.txt"
#for within docker image
strControlFile = "/mnt/pythoncode/controlaction.txt"

dControlData = { "Action": "nothing", "AttackNumber": 300 }




sCommands ="(a)ttack,capture,capturepet,(d)etectpet,exit,processfile or ctlr-c':"
print(sCommands)
command = ""
previousCommand = ""
while str(command) != 'wow':
    
    if previousCommand == "detectpet":
        time.sleep(5)
        Command = "detectpet"
    else:
        command = raw_input(">")#for widows, use input.
    if command == "a":
        command = "attack"
    if command == "processfile":
        Processfile()
        command = previousCommand
    if command == "detectpet" or command == "d":
        command = "detectpet"
    if command == "getgold" or command == "g":
        pass    
    dControlData['Action'] = str(command)
    controlFile = open( strControlFile, "wb" )
    pickle.dump( dControlData, controlFile ,protocol=2)
    controlFile.close()
    if command == "detectpet":
        time.sleep(5)
        prediction, filecounts = predictpet(filecounts)
        if prediction == 0:
            dControlData['Action'] = str("getgold")
            controlFile = open( strControlFile, "wb" )
            pickle.dump( dControlData, controlFile ,protocol=2)
            controlFile.close()
            print("ready to get gold")

    previousCommand = command

    




