import os
import time
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import pickle

imagenumber = 1


def captureImage(imagenumber):
    start = time.time()
    # tupSquare=(500,500,10,10)
    # result = device.takeSnapshot()#.getSubImage(tupSquare)
    # Writes the screenshot to a file
    # s="C:\Users\TGDLOHU1\Downloads\imagecaptures\shotnm"+str(imagenumber) + ".png"
    # s2="adb pull /sdcard/foo2.png " + "C:\Users\TGDLOHU1\Downloads\imagecaptures\shot2nm"+str(i) + ".png"
    # sfile="/sdcard/snapshot/c" + str(i) + ".png
    sfile = "/sdcard/windows/BstSharedFolder/c" + str(imagenumber) + ".raw"
    # sshell="screencap -p > " + sfile
    sshell = "screencap  > " + sfile
    device.shell(sshell)
    # print sshell
    # sdir=" C:\\Users\\TGDLOHU1\\Downloads\\imagecaptures\\"
    # print sdir
    # spull="adb pull " + sfile + sdir
    # result.writeToFile(s,'png')
    end = time.time()
    print "captureImage time: ", end - start
    return


def capturePetMode(imagenumber):
    start = time.time()
    # tupSquare=(624,364,110,110)
    result = device.takeSnapshot()  # .getSubImage(tupSquare)
    # Writes the screenshot to a file
    s = "C:\Users\TGDLOHU1\Downloads\imagecaptures\checkpet" + str(imagenumber) + ".png"
    # s2="adb pull /sdcard/foo2.png " + "C:\Users\TGDLOHU1\Downloads\imagecaptures\shot2nm"+str(i) + ".png"
    # sfile="/sdcard/snapshot/c" + str(i) + ".png"
    # sshell="screencap -p > " + sfile
    # print sshell
    # sdir=" C:\\Users\\TGDLOHU1\\Downloads\\imagecaptures\\"
    # print sdir
    # spull="adb pull " + sfile + sdir
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
    # s2="adb pull /sdcard/foo2.png " + "C:\Users\TGDLOHU1\Downloads\imagecaptures\shot2nm"+str(i) + ".png"
    # sfile="/sdcard/snapshot/c" + str(i) + ".png"
    # sshell="screencap -p > " + sfile
    # print sshell
    # sdir=" C:\\Users\\TGDLOHU1\\Downloads\\imagecaptures\\"
    # print sdir
    # spull="adb pull " + sfile + sdir
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
    dControlData['Action'] = "nothing"
    pickle.dump(dControlData, file, protocol=2)
    file.close()
    end = time.time()
    print "resetfile: ", end - start
    return


print("jPython controler on")
device = MonkeyRunner.waitForConnection()
print("connected")
dControlData = {}

strControlFile = "C:\\Users\\TGDLOHU1\\pythoncode\\controlaction.txt"

# print (oControlData.action)
# controlFile = open( strControlFile, "wb" )
# pickle.dump( oControlData, controlFile )
# controlFile.close()


controlFile = open(strControlFile, "rb")
dControlData = pickle.load(controlFile)
print (dControlData["Action"])
controlFile.close()

strAction = "nothing"
while strAction != "Exit":
    controlFile = open(strControlFile, "rb")
    dControlData = pickle.load(controlFile)
    strAction = dControlData["Action"]
    controlFile.close()
    MonkeyRunner.sleep(0.51)
    # print(charControl)
    # end = time.time()
    # print "cycletime: ", end - start
    if strAction == "attack":
        hitautomatically(dControlData["AttackNumber"])
    if strAction == "capture":
        captureImage(1)
        resetfile(strControlFile, dControlData)
    if strAction == "capturepet":
        capturePetMode(imagenumber)
        imagenumber += 1
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







# os.system("C:\Users\TGDLOHU1\AppData\Local\Programs\Python\Python35\python C:\Users\TGDLOHU1\Downloads\example2.py")
