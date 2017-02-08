from __future__ import print_function


class CurrentStatus:
    """holds the game current status """

    def __init__(self):
        self.level = 1
        self.current_hat = "unknown"
        self.current_pet = "unknown"
        self.current_hero = "unknown"
        self.current_tab = "unknown"

    def update_status(self, pred_dict, trainers_predictors_list):
        if "main_level" in pred_dict:
            self.update_level((pred_dict["previous_level"], pred_dict["main_level"], pred_dict["next_level"]))
        if "tab_predictor" in pred_dict:
            for class_predictor in trainers_predictors_list:
                if class_predictor.name == "tab_predictor":
                    self.current_tab = class_predictor.pred_classes[pred_dict["tab_predictor"]]
        self.show_status()

    def update_level(self, level_tuple):
        print("last:", level_tuple[0], "curr:", level_tuple[1], "next:", level_tuple[2])
        current_level = level_tuple[1]
        last_level = level_tuple[0]
        next_level = level_tuple[2]

        if current_level.isdigit() and last_level.isdigit():
            if int(current_level) == int(last_level) + 1:
                self.level = int(current_level)
        if current_level.isdigit() and next_level.isdigit():
            if int(current_level) == int(next_level) - 1:
                self.level = int(current_level)
        if last_level.isdigit() and next_level.isdigit():
            if int(last_level) == int(next_level) - 2:
                self.level = int(last_level) + 1
        if current_level.isdigit() and int(current_level) >= int(self.level):
            if int(current_level) < 9999:
                self.level = int(current_level)

    def show_status(self):
        print("Status: level: ", str(self.level), " tab: ", self.current_tab )
