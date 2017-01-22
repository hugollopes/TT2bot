import globals as glo
import pickle
import time


def capture_gold_forever(predictor):
    while True:
        insert_command("capture")
        acknowledge()
        pred_dict = predictor.predict()
        if predictor.check_predict(pred_dict, 'gold_pet_predictor', "goldpet"):
            if predictor.check_predict(pred_dict, 'boss_active_predictor', "boss_inactive"):
                insert_command("hit", hit_pos="boss_toggle")
                print("ready to get gold with boss")
                time.sleep(0.2)
                insert_command("hit", hit_pos="pet_gold_hit_center")
                insert_command("hit", hit_pos="pet_gold_hit_normal")
                acknowledge()
            else:
                insert_command("hit", hit_pos="pet_gold_hit_center")
                insert_command("hit", hit_pos="pet_gold_hit_normal")
                print("ready to get gold")
                acknowledge()
            upgrade_heroes()


def upgrade_heroes():
    #insert_command("hit", hit_pos="heroes_tab") #must assume it is in the hero tab
    insert_command("drag", start_tuple=(296, 1179), end_tuple=(293, 833), duration=0.5, steps=10)
    insert_command("hit", hit_pos="last_hero_upg") #assumes well possitioned at the bottom of the heroes tab.
    insert_command("hit", hit_pos="before_last_hero_upg")
    insert_command("hit", hit_pos="2_before_last_hero_upg")
    insert_command("hit", hit_pos="3_before_last_hero_upg")
    acknowledge()


def acknowledge():
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
                if key == "hit_pos":
                    action["X"] = glo.HIT_DICT[value][0]
                    action["Y"] = glo.HIT_DICT[value][1]
                else:
                    action[key] = value
                    # print("%s == %s" % (key, value))
        command_list.insert(0, action)
        control_data["ActionList"] = command_list
        control_file = open(glo.CONTROL_FILE, "wb")
        pickle.dump(control_data, control_file, protocol=2)
        control_file.close()
    except Exception:
        print("some error inserting command. action=nothing")
