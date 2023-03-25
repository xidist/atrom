import os
import config.config
import json
import torchaudio

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



def get_wav_and_pitch_intervals_for_file(file_path: str, sample_rate = 16000):
    """
    file_path: the path of a training file in the maestro dataset

    returns: a tuple of a 1D Tensor containing the PCM data, 
             and a list of three-element tuples from the corresponding midi file
    """
    return (get_wave_for_file(file_path, sample_rate=sample_rate),
            get_pitch_intervals_for_file(file_path))

def get_wav_for_file(file_path: str, sample_rate=16000):
    """
    file_path: the path of a training file in the maestro dataset

    returns: a 1D Tensor containing the PCM data
    """
    
    info = torchaudio.info(file_path)
    if not (info.sample_rate == 44100 or info.sample_rate == 48000):
        raise Exception(f"{file_path} failed sample rate sanity check: {info.sample_rate}")
    if not (info.num_channels == 1 or info.num_channels == 2):
        raise Exception(f"{file_path} failed channel count sanity check: {info.num_channels}")
    
    waveform, file_sample_rate = torchaudio.load(file_path)
    if info.num_channels == 2:
        waveform = torch.sum(waveform, dim=0)

    # squeeze out the channel dimension, since it's already mono
    waveform = waveform.squeeze()
    waveform = torchaudio.functional.resample(waveform, file_sample_rate, sample_rate)
    return waveform
    

def get_pitch_intervals_for_file(file_path: str):
    """
    file_path: the path to a training file in the maestro dataset

    returns: a list of three-element tuples representing the pitch, start time, and end time of notes
    """
    
    base = os.path.splitext(file_path)[0]
    file_path = base + ".txt"
    with open(file_path) as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]
