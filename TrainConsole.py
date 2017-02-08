from __future__ import print_function
from genericTrain import TT2Predictor , TrainerPredictor
import globals as glo


from PIL import Image
import tesserocr
print (tesserocr.tesseract_version() )
print(tesserocr.get_languages())
images = [glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER + '/fullcapture961 .png']

#with PyTessBaseAPI() as api:
#    for img in images:
#        api.SetImageFile(img)
#        print( api.GetUTF8Text())
#        print( api.AllWordConfidences())

img = Image.open(glo.DATA_FOLDER + '/number_range_predictorcropped3.png')#glo.UNCLASSIFIED_GLOBAL_CAPTURES_FOLDER + '/fullcapture961 .png')
#img = img.convert('L')

from tesserocr import PyTessBaseAPI, RIL, iterate_level,PSM
#print(help(tesserocr))

api = PyTessBaseAPI()
api.Init()
api.SetImageFile(glo.DATA_FOLDER + '/number_range_predictorcropped3.png')
api.SetVariable("tessedit_pageseg_mode", "7")
api.SetVariable("language_model_penalty_non_dict_word","0")
api.SetVariable("doc_dict_enable", "0")
print("recognized txt:",api.GetUTF8Text().encode('utf-8').strip())
#api.Recognize()
"""
ri = api.GetIterator()
level = RIL.SYMBOL
for r in iterate_level(ri, level):
    symbol = r.GetUTF8Text(level)  # r == ri
    conf = r.Confidence(level)
    print(u'symbol {}, conf: {}'.format(symbol, conf).encode('utf-8').strip())
    indent = False
    ci = r.GetChoiceIterator()
    for c in ci:
        if indent:
            print('\t\t ',)
        print('\t- ',)
        choice = c.GetUTF8Text()  # c == ci
        print(            u'{} conf: {}'.format(choice, c.Confidence()).encode('utf-8').strip())
        indent = True


    #print("aquiiii", tesserocr.image_to_text(img))
"""

predictor = TT2Predictor()
print("creating new trainer")
new_predictor = TrainerPredictor("tab_predictor", ["skills_tab", "heroes_tab", "equipment_tab",
                                                           "pet_tab", "relic_tab", "shop_tab", "no_tab"]
                                         , (51, 1, 59, 717)
                                         , 2, 179, 255.0
                                         , [200, 30])

print("trainer created")
#todo: need a number by time on created crops or copying the original name.
#new_predictor.crop_images(selected_globals=True)
#new_predictor.process_images()
#new_predictor.read_and_pickle()
#new_predictor.train_graph()
"""

predictor.parse_raw_image()
predictor.predict_parsed_all()
"""