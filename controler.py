import time
import os
from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import pickle







def captureImage():
    start = time.time()
    file = "/sdcard/windows/BstSharedFolder/c1.raw"
    sshell = "screencap  > " + file
    device.shell(sshell)
    end = time.time()
    print "captureImage time: ", end - start
    return


def hit(x, y):
    device.touch(int(x), int(y), MonkeyDevice.DOWN_AND_UP)
    MonkeyRunner.sleep(0.03)


def hitautomatically(times):
    start = time.time()
    for i in range(1, times + 1):
        device.touch(350, 500, MonkeyDevice.DOWN_AND_UP)
        MonkeyRunner.sleep(0.03)
    end = time.time()
    print "hitautomaticallytime: ", end - start
    return


def resetfile(strFile):
    start = time.time()
    file = open(strFile, "wb")
    dControlData = {"ActionList": [{"Type": "nothing"}]}
    pickle.dump(dControlData, file, protocol=2)
    file.close()
    end = time.time()
    print "resetfile: ", end - start
    return


def write_done():
    acknowledge_file = "C:\\ProgramData\\Bluestacks\\UserData\\SharedFolder\\ackaction.txt"
    control_file = open(acknowledge_file, "wb")
    control_data = {"Done": "yes"}
    pickle.dump(control_data, control_file, protocol=2)
    control_file.close()

def read_action(control_file_path):
    try:
        control_file = open(control_file_path, "r+")
        control_data = pickle.load(control_file)
        control_file.close()
        command_list = control_data["ActionList"]
        #print "control_data",control_data
        if not command_list:
            write_done()
            command_list = [{"Type": "nothing"}]
        action = command_list.pop()
        control_data["ActionList"] = command_list
        #print "control_data", control_data
        control_file = open(control_file_path, "wb")
        pickle.dump(control_data, control_file, protocol=2)
        control_file.close()
    except Exception:
        print("some erro reading file. action=nothing", str(Exception))
        action = {"Type": "nothing"}
    return action


def get_gold():
    print("getgold")
    # hit center
    device.touch(360, 562, MonkeyDevice.DOWN_AND_UP)
    MonkeyRunner.sleep(0.03)
    # hit lowered pet
    device.touch(427, 614, MonkeyDevice.DOWN_AND_UP)

print("jPython controller on")
device = MonkeyRunner.waitForConnection()
print("connected")

image_number = 1
strControlFile = "C:\\ProgramData\\Bluestacks\\UserData\\SharedFolder\\controlaction.txt"
resetfile(strControlFile)

strAction = "nothing"
while strAction != "Exit":
    MonkeyRunner.sleep(0.1)
    action = read_action(strControlFile)
    strAction = action["Type"]
    if strAction == "attack":
        hitautomatically(300)
    if strAction == "capture":
        captureImage()
    if strAction == "hit":
        hit(action["X"],action["Y"])
    if strAction == "getgold":
        get_gold()

