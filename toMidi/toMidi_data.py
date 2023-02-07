import os
import config.config
import json

def load_maestro_json():
    maestro_dir = config.config.read_config_value(["toMidi", "maestro"])
    with open(maestro_dir + "/maestro-v3.0.0.json") as f:
        data = json.load(f)
        # json object with keys:
        #   ['canonical_composer', 'canonical_title', 'split', 'year', 'midi_filename', 'audio_filename', 'duration']
        # each value for a key is a dictionary of
        #   unique ids as examples
        #   the value of the outer key as the value
    return data
        

def get_training_files():
    """
    Get the list of names of files to use for training
    
    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """
    data = load_maestro_json()
    train_ids = [k for k, v in data["split"].items() if v == "train"]
    filenames = [data["audio_filename"][k] for k in train_ids]
    filenames = [config.config.read_config_value(["toMidi", "maestro"]) + "/" + f for f in filenames]
    return filenames


def get_validation_files():
    """
    Get the list of names of files to use for validation
    
    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """
    data = load_maestro_json()
    validate_ids = [k for k, v in data["split"].items() if v == "validation"]
    filenames = [data["audio_filename"][k] for k in validate_ids]
    filenames = [config.config.read_config_value(["toMidi", "maestro"]) + "/" + f for f in filenames]
    return filenames


def get_test_files():
    """
    Get the list of names of files to use for testing
    
    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """
    data = load_maestro_json()
    test_ids = [k for k, v in data["split"].items() if v == "test"]
    filenames = [data["audio_filename"][k] for k in test_ids]
    filenames = [config.config.read_config_value(["toMidi", "maestro"]) + "/" + f for f in filenames]
    return filenames



