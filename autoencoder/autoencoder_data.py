import torch
import torchaudio
import os
import config.config


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


def get_training_files():
    """
    Get the list of names of files to use for training

    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """
    return config.config.read_data_file(["autoencoder", "train"])


def get_validation_files():
    """
    Get the list of names of files to use for validation

    returns: list[string]. each path should be either an absolute path,
             or relative to the current working directory
    """
    return config.config.read_data_file(["autoencoder", "validation"])


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

    return config.config.read_data_file(["autoencoder", "demo"])


def get_checkpoint_file_path():
    """,
    Get the name of the file to use for saving and resuming training
 
    returns: string. the path should be either an absolute path,
             or relative to the current working directory
    """

    return config.config.read_config_value(["autoencoder", "checkpoint"])


def get_demo_write_directory():
    """
    Get the name of the directory to put reconstructed demo files in. (Demo files
    will be put into subdirectories based on the epoch after they were created)

    returns: string. the path should be either an absolute path,
             or relative to the current working directory
    """

    return config.config.read_config_value(["autoencoder", "demo_write"])






