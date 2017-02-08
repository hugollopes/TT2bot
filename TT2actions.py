from __future__ import print_function
import globals as glo
import pickle
import time


def play(predictor):
    count_check_egg = 0
    count_upgrade_all_heroes = 0
    count = 0
    while True:
        print("run:", count)
        capture_gold(predictor)
        upgrade_heroes(predictor)
        insert_command("attack")
        insert_command("attack")
        insert_command("attack")
        insert_command("attack")
        insert_command("attack")
        acknowledge()
        count += 1
        count_upgrade_all_heroes += 1
        count_check_egg += 1
        if count_check_egg >= 100:
            recognize_and_get_egg(predictor, False)
            count_check_egg = 0
        if count_upgrade_all_heroes >= 30:
            upgrade_all_heroes(predictor)
            count_upgrade_all_heroes = 0


def capture_gold(predictor):
    insert_command("capture")
    acknowledge()
    predictor.parse_raw_image()
    pred_dict = predictor.predict_parsed(["gold_pet_predictor"], ["previous_level", "main_level", "next_level"])
    if predictor.check_predict(pred_dict, 'gold_pet_predictor', "goldpet"):
        pred_dict = predictor.predict_parsed(["boss_active_predictor"], [])
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


def capture_gold_forever(predictor):
    while True:
        capture_gold(predictor)


def recognize_and_get_egg(predictor, capture):
    if capture:
        insert_command("capture")
        acknowledge()
        predictor.parse_raw_image()
    start = time.time()
    pred_dict = predictor.predict_parsed(["egg_active_predictor"], ["previous_level", "main_level", "next_level", "last_hero"])
    print("prediction time: ", time.time() - start)
    if int(pred_dict['egg_active_predictor']) == 0:
        print("capturing egg")
        insert_command("hit", hit_pos="egg")
        time.sleep(0.5)
        insert_command("hit", hit_pos="Shinning_egg")
        time.sleep(0.5)
        insert_command("hit", hit_pos="Shinning_egg")
        time.sleep(1)
        insert_command("hit", hit_pos="Shinning_egg")  # one more hit to clear
        time.sleep(3)
        insert_command("hit", hit_pos="Shinning_egg")  # one more hit to clear
        acknowledge()


def upgrade_heroes(predictor):
    go_to_heroes_tab(predictor)
    #insert_command("hit", hit_pos="heroes_tab") #must assume it is in the hero tab
    #insert_command("drag", start_tuple=(296, 1179), end_tuple=(293, 833), duration=0.5, steps=10)
    insert_command("drag", drag_pos="drag_down_a_lot")
    #insert_command("hit", hit_pos="last_hero_upg") #assumes well possitioned at the bottom of the heroes tab.
    insert_command("hit", hit_pos="before_last_hero_upg")
    #insert_command("hit", hit_pos="2_before_last_hero_upg")
    #insert_command("hit", hit_pos="3_before_last_hero_upg")
    acknowledge()


def upgrade_all_heroes(predictor):
    go_to_heroes_tab(predictor)
    for _ in range(10):
        insert_command("drag", drag_pos="drag_down_a_lot")
    first = False  # meaning we are in the first page
    heroes = []
    for _ in range(20):
        insert_command("capture")
        acknowledge()
        predictor.parse_raw_image()
        pred_dict = predictor.predict_parsed([], ["previous_level", "main_level", "next_level", "last_hero"])
        heroes.append(pred_dict["last_hero"])
        if pred_dict["last_hero"] == "Lance, Knight of Cobalt Steel":
            first = True
        insert_command("hit", hit_pos="last_hero_upg") #assumes well possitioned at the bottom of the heroes tab.
        insert_command("hit", hit_pos="last_hero_upg")
        insert_command("hit", hit_pos="last_hero_upg")
        insert_command("hit", hit_pos="last_hero_upg")
        insert_command("hit", hit_pos="last_hero_upg")
        insert_command("hit", hit_pos="last_hero_upg")
        acknowledge()
        insert_command("drag", drag_pos="drag_1_hero_up")
        acknowledge()
    print(heroes)


def capture_clan_boss(predictor):
    insert_command("hit", hit_pos="hit_clan")
    acknowledge()
    time.sleep(2)
    insert_command("hit", hit_pos="hit_clan_quest")
    acknowledge()
    time.sleep(4)
    insert_command("hit", hit_pos="boss_fight")
    acknowledge()
    time.sleep(4)
    insert_command("attack")
    insert_command("attack")
    insert_command("attack")
    insert_command("attack")
    time.sleep(4)
    insert_command("hit", hit_pos="close_clan_quest")
    acknowledge()
    time.sleep(1)
    insert_command("hit", hit_pos="close_clan")
    acknowledge()
    time.sleep(1)


def go_to_heroes_tab(predictor):
    insert_command("capture")
    acknowledge()
    # start = time.time()
    predictor.parse_raw_image()
    pred_dict = predictor.predict_parsed(["tab_predictor"], [])
    if predictor.check_predict(pred_dict, 'tab_predictor', "heroes_tab"):
        pass
    else:
        insert_command("hit", hit_pos="heroes_tab")
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
                if key == "drag_pos":
                    action["start_tuple"] = glo.DRAG_DICT[value][0]
                    action["end_tuple"] = glo.DRAG_DICT[value][1]
                    action["duration"] = glo.DRAG_DICT[value][2]
                    action["steps"] = glo.DRAG_DICT[value][3]
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
