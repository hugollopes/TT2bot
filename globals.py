def set_globals():
    _data_folder = "/mnt/tt2botdata"
    _share_folder = "/mnt/directShare"
    _code_folder = "/mnt/pythoncode"
    _pet_prediction_samples_folder = _data_folder + "/dataforclassifier/TT2predictionsamples"
    _pet_classified_data_folder = _data_folder + "/dataforclassifier/TT2classified"
    _unclassified_global_captures_folder = _data_folder + "/dataforclassifier/unclassified/globalcaptures"
    _control_file = _share_folder + "/controlaction.txt"
    _raw_full_file = _share_folder + "/c1.raw"
    return _data_folder, _share_folder, _code_folder, _pet_prediction_samples_folder, _pet_classified_data_folder, _unclassified_global_captures_folder, _control_file, _raw_full_file


