import torch
import torchaudio
import os
from config.config import *


def load_and_check(file_path, hp):
    """
    file_path: str, the path to the audio file to load
    hp: Hyperparameters

    returns: a tuple of (signal, sample_rate)
    signal: a Tensor[sample_length] containing the audio data, in the range -1 to 1
    sample_rate: int, the number of samples recorded each second

    hp: Hyperparameters
    """

    # lets do some basic sanity checks on the file format
    # these should hold on maestro, but might be too strict for youtube/JMS?
    # if so, we'll have to think about how to handle the failures
    info = torchaudio.info(file_path)

    if not (info.sample_rate == 44100 or info.sample_rate == 48000):
        raise Exception(f"{file_path} failed sample rate sanity check: {info.sample_rate}")
    if not (info.num_channels == 1 or info.num_channels == 2):
        raise Exception(f"{file_path} failed channel count sanity check: {info.num_channels}")
    
    waveform, sample_rate = torchaudio.load(file_path)

    if info.num_channels == 2:
        waveform = torch.sum(waveform, dim=0)

    # squeeze out the channel dimension, since it's already mono
    waveform = waveform.squeeze()
    waveform = waveform.to(hp.device)

    if info.sample_rate != hp.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, hp.sample_rate)
    
    return waveform, sample_rate



def recursively_find_files_in_dir(dir):
    """
    dir: string. The root directory to search at
    
    returns: list[string]. Each path starts with `dir`
    """

    result = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            result.append(os.path.join(dirpath, file))
    return result

def is_wav_file(filename):
    """
    filename: string

    returns: bool. True if the file has a .wav extension, and False otherwise
    """
    return os.path.splitext(filename)[1] == ".wav"

ENABLE_MADDY_LOCAL_TESTING = False

def get_training_files():
    """
    Get the list of names of files to use for training

    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """

    if ENABLE_MADDY_LOCAL_TESTING:
        # mark: local maddy testing. remove in the future
        search_dir = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio"
        files = ["lig_orchestra.wav", "lig_vocals.wav", "myshot.wav"]
        return [os.path.join(search_dir, f) for f in files]

    search_dir = "/z/atrom/datasets/unlabeled/YouTube"
    wav_files = [f for f in recursively_find_files_in_dir(search_dir)
                 if is_wav_file(f)]

    return wav_files[:int(len(wav_files) * 7 / 10)]
    
def get_validation_files():
    """
    Get the list of names of files to use for validation

    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """

    if ENABLE_MADDY_LOCAL_TESTING:
        # mark: local maddy testing. remove in the future
        search_dir = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio"
        files = ["lig_soundtrack.wav", "all-star.wav"]
        return [os.path.join(search_dir, f) for f in files]
    
    search_dir = "/z/atrom/datasets/unlabeled/YouTube"
    wav_files = [f for f in recursively_find_files_in_dir(search_dir)
                 if is_wav_file(f)]

    return wav_files[int(len(wav_files) * 7 / 10) :
                     int(len(wav_files) * 85 / 100)]


def get_demo_files():
    """
    Get the list of names of files to use for demoing. Every so often,
    the program will pass each demo file through the autoencoder,
    saving the output audio to disk (i.e. for humans to qualitatively
    assess training progress). For the most accurate assessment,
    only use files in the validation set. 

    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """
    
    if ENABLE_MADDY_LOCAL_TESTING:
        # mark: local maddy testing. remove in the future
        return get_training_files() + get_validation_files()

    val = get_validation_files()
    return val[::int(len(val) / 20)]

def get_checkpoint_file_path():
    """,
    Get the name of the file to use for saving and resuming training
 
    returns: string. the path should be either an absolute path,
             or relative to the current working directory
    """

    if ENABLE_MADDY_LOCAL_TESTING:
        # mark: local maddy testing. remove in the future
        return "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/checkpoint"
    
    return "/z/atrom/autoencoder_checkpoint"

def get_demo_write_directory():
    """
    Get the name of the directory to put reconstructed demo files in. (Demo files
    will be put into subdirectories based on the epoch after they were created)

    returns: string. the path should be either an absolute path,
             or relative to the current working directory
    """

    if ENABLE_MADDY_LOCAL_TESTING:
        # mark: local maddy testing. remove in the future
        return "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/output_audio"

    
    return "/z/atrom/demo-files"






