import globals as glo
import pickle
import time


def capture_gold_forever(predictor):
    while True:
        reset_acknowledge()
        insert_command("capture")
        wait_acknowledge()
        predictor.parse_raw_image()
        pred_dict = predictor.predict_parsed_all()
        if int(pred_dict['gold_pet_predictor']) == 0:
            if int(pred_dict['boss_active_predictor']) == 1:
                insert_command("hit", X=glo.HIT_DICT["boss_toggle"][0], Y=glo.HIT_DICT["boss_toggle"][1])
                print("ready to get gold with boss")
                time.sleep(0.5)
                insert_command("getgold")  # todo: remove gold for generic hit.
                reset_acknowledge()
                wait_acknowledge()
                time.sleep(0.5)
            else:
                insert_command("getgold")  # todo: remove gold for generic hit.
                print("ready to get gold")
                reset_acknowledge()
                wait_acknowledge()


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


def insert_command(_command, **kwargs):
    try:
        control_file = open(glo.CONTROL_FILE, "rb")
        control_data = pickle.load(control_file)
        control_file.close()
        command_list = control_data["ActionList"]
        action = {"Type": _command}
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                action[key] = value
                # print("%s == %s" % (key, value))
        command_list.insert(0, action)
        control_data["ActionList"] = command_list
        control_file = open(glo.CONTROL_FILE, "wb")
        pickle.dump(control_data, control_file, protocol=2)
        control_file.close()
    except Exception:
        print("some error inserting command. action=nothing")
