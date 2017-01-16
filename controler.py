import time
import os
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import pickle







def captureImage(image_number_count):
    start = time.time()
    file = "/sdcard/windows/BstSharedFolder/c" + str(image_number_count) + ".raw"
    sshell = "screencap  > " + file
    device.shell(sshell)
    end = time.time()
    print "captureImage time: ", end - start
    return


def capturePetMode(image_number_count):
    start = time.time()
    # tupSquare=(624,364,110,110)
    result = device.takeSnapshot()  # .getSubImage(tupSquare)
    # Writes the screenshot to a file
    s = "C:\Users\TGDLOHU1\Downloads\imagecaptures\checkpet" + str(image_number_count) + ".png"
    result.writeToFile(s, 'png')
    end = time.time()
    print "captureImage time: ", end - start
    return


def detectPetMode():
    start = time.time()
    tupSquare = (624, 364, 110, 110)
    result = device.takeSnapshot().getSubImage(tupSquare)
    # Writes the screenshot to a file
    s = "C:\Users\TGDLOHU1\pythoncode\detect.png"
    result.writeToFile(s, 'png')
    end = time.time()
    print "captureImage time: ", end - start
    return


def hitautomatically(times):
    start = time.time()
    for i in range(1, times + 1):
        device.touch(350, 500, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
    end = time.time()
    print "hitautomaticallytime: ", end - start
    return


def resetfile(strFile, dControlData):
    start = time.time()
    file = open(strFile, "wb")
    dControlData['Action'] = ["nothing"]
    pickle.dump(dControlData, file, protocol=2)
    file.close()
    end = time.time()
    print "resetfile: ", end - start
    return


def read_action(control_file_path):
    try:
        control_file = open(control_file_path, "r+")
        control_data = pickle.load(control_file)
        control_file.close()
        command_list = control_data["Action"]
        #print "control_data",control_data
        if not command_list:
            command_list = ["nothing"]
        action = command_list.pop()
        control_data["Action"] = command_list
        #print "control_data", control_data
        control_file = open(control_file_path, "wb")
        pickle.dump(control_data, control_file, protocol=2)
        control_file.close()
    except Exception:
        print("some erro reading file. action=nothing" , str(Exception))
        action = "nothing"
    return action

print("jPython controler on")
device = MonkeyRunner.waitForConnection()
print("connected")

image_number = 1
dControlData = {"Action":["nothing"],"AttackNumber":300}
strControlFile = "C:\\ProgramData\\Bluestacks\\UserData\\SharedFolder\\controlaction.txt"
resetfile(strControlFile,dControlData)

strAction = "nothing"
while strAction != "Exit":
    MonkeyRunner.sleep(1)
    strAction = read_action(strControlFile)
    #if strAction != "nothing":
    print "action",strAction
    if strAction == "attack":
        hitautomatically(300)
    if strAction == "capture":
        captureImage(1)
        resetfile(strControlFile, dControlData)
    if strAction == "capturepet":
        capturePetMode(image_number)
        image_number += 1
    if strAction == "detectpet":
        detectPetMode()
        resetfile(strControlFile, dControlData)
    if strAction == "getgold":
        print("getgold")
        # hit center
        device.touch(360, 562, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
        device.touch(360, 562, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
        device.touch(360, 562, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
        # hit lowered pet
        device.touch(427, 614, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
        device.touch(427, 614, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
        device.touch(427, 614, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
        resetfile(strControlFile, dControlData)

