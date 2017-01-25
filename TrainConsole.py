from __future__ import print_function
from genericTrain import TT2Predictor , TrainerPredictor



predictor = TT2Predictor()
print("creating new trainer")
new_predictor = TrainerPredictor("tab_predictor", ["skills_tab", "heroes_tab", "equipment_tab",
                                                   "pet_tab", "relic_tab", "shop_tab", "no_tab"]
                                            , (51, 1, 59, 717)
                                            , 2, 179, 255.0
                                            , [200, 30])
print("trainer created")
#todo: need a capture and parse
#new_predictor.crop_images(selected_globals=True)
new_predictor.process_images()
new_predictor.read_and_pickle()
new_predictor.train_graph()
"""

predictor.parse_raw_image()
predictor.predict_parsed_all()
"""