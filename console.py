import pickle
import sys
import os
import time
from PIL import Image
sys.path.append(os.path.abspath("/mnt/pythoncode"))
from train import Processfile,predictpet

#this function returns the total increment of classified and unclassified pet pictures
def getTotalNumberFiles():
    totalnumber = 0
    filepaths = ["/mnt/pythoncode/dataforclassifier/TT2predictionsamples/goldpet",
        "/mnt/pythoncode/dataforclassifier/TT2predictionsamples/nopet",
        "/mnt/pythoncode/dataforclassifier/TT2predictionsamples/normalpet",
        "/mnt/pythoncode/dataforclassifier/TT2predictionsamples/partial pet",
        "/mnt/pythoncode/dataforclassifier/TT2classified/goldpet",
        "/mnt/pythoncode/dataforclassifier/TT2classified/nopet",
        "/mnt/pythoncode/dataforclassifier/TT2classified/normalpet",
        "/mnt/pythoncode/dataforclassifier/TT2classified/partial pet",
        "/mnt/pythoncode/dataforclassifier/unclassified/globalcaptures"]
    for path in filepaths:
        src_files = os.listdir(path)
        for file_name in src_files:
            totalnumber +=1
    return totalnumber
            

def parseRawImage(totalnumber,saveoriginal):
    start = time.time()
    #test raw reading
    iImgSize= 40
    with open('/mnt/directShare/c1.raw', 'rb') as f:
        im = Image.frombytes('RGBA', (1280,720), f.read())
    petcrop = im.crop((624,364,624+110,364+110))
    petcrop.save('/mnt/pythoncode/detect.png')
    petcrop = petcrop.convert('L')
    imageTuple = (iImgSize,iImgSize)
    petcrop = petcrop.resize(imageTuple)
    petcrop.save('/mnt/pythoncode/detectResized.png')
    if saveoriginal:
        im.save("/mnt/pythoncode/dataforclassifier/unclassified/globalcaptures/fullcapture" + str (totalnumber) +" .png")
    end = time.time()
    print("captureImage time: ", end - start)



totalnumber = getTotalNumberFiles()
print("there are total ", totalnumber ,"files")
dControlData = {}

#for windows: #strControlFile = "C:\\Users\\TGDLOHU1\\pythoncode\\controlaction.txt"
#for within docker image
strControlFile = "/mnt/pythoncode/controlaction.txt"

dControlData = { "Action": "nothing", "AttackNumber": 300 }




sCommands ="(a)ttack,capture,capturepet,(d)etectpet,exit,(cg)captureforgold,processfile or ctlr-c':"
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
    if command == "captureforgold" or command == "cg":
        forever = True
        while forever:
            dControlData['Action'] = str("capture")
            controlFile = open( strControlFile, "wb" )
            pickle.dump( dControlData, controlFile ,protocol=2)
            controlFile.close()
            
            time.sleep(2) #wait for android processing to optaing the raw image
            parseRawImage(totalnumber,False)
            time.sleep(0.3)
            prediction, totalnumber = predictpet(totalnumber)
            if prediction == 0:
                dControlData['Action'] = str("getgold")
                controlFile = open( strControlFile, "wb" )
                pickle.dump( dControlData, controlFile ,protocol=2)
                controlFile.close()
                print("ready to get gold")
                time.sleep(1) #wait for not overwritten
        
        
    dControlData['Action'] = str(command)
    controlFile = open( strControlFile, "wb" )
    pickle.dump( dControlData, controlFile ,protocol=2)
    controlFile.close()
    if command == "detectpet":
        time.sleep(5)
        prediction, totalnumber = predictpet(totalnumber)
        if prediction == 0:
            dControlData['Action'] = str("getgold")
            controlFile = open( strControlFile, "wb" )
            pickle.dump( dControlData, controlFile ,protocol=2)
            controlFile.close()
            print("ready to get gold")

    previousCommand = command

    




