DATA_FOLDER = "/mnt/tt2botdata"
SHARE_FOLDER = "/mnt/directShare"
code_folder = "/mnt/pythoncode"
PET_PREDICTION_SAMPLES_FOLDER = DATA_FOLDER + "/dataforclassifier/TT2predictionsamples"
PET_CLASSIFIED_DATA_FOLDER = DATA_FOLDER + "/dataforclassifier/TT2classified"
UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER = DATA_FOLDER + "/dataforclassifier/unclassified/globalcaptures"
SELECTED_CAPTURES_FOLDER = DATA_FOLDER + "/dataforclassifier/unclassified/selected_globals"
CONTROL_FILE = SHARE_FOLDER + "/controlaction.txt"
ACKNOWLEDGE_FILE = SHARE_FOLDER + "/ackaction.txt"
RAW_FULL_FILE = SHARE_FOLDER + "/c1.raw"
HIT_DICT = {"egg": (50, 525),
            "Shinning_egg": (370, 355),
            "boss_toggle": (605, 49),
            "pet_gold_hit_center": (360, 562),
            "pet_gold_hit_normal": (427, 614),
            "heroes_tab": (178, 1258),
            "last_hero_upg": (606, 1172),
            "before_last_hero_upg": (606, 1062),
            "2_before_last_hero_upg": (606, 943),
            "3_before_last_hero_upg": (606, 835),
            "hit_clan": (122, 34),
            "hit_clan_quest": (135, 1164),
            "close_clan": (642, 69),
            "close_clan_quest": (640, 70),
            "boss_fight": (459, 1175)}
DRAG_DICT = {"drag_down_a_lot": ((296, 1179), (293, 833), 0.5, 10),
             "drag_1_hero_down": ((105, 1127), (105, 1007), 0.01, 1000),
             "drag_1_hero_up": ((105, 1007), (105, 1127), 0.01, 1000)}
