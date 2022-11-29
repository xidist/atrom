import torchaudio
import math
import os

torch.manual_seed(0)
random.seed(0)
print("finished importing modules...")

def check(file_path):
    """
    Performs the same checks as load_and_check from autoencoder.py

    file_path: str, the path to the audio file to check. 

    returns: float. the duration of the file, in seconds
    """

    # lets do some basic sanity checks on the file format
    # these should hold on maestro, but might be too strict for youtube/JMS?
    # if so, we'll have to think about how to handle the failures
    info = torchaudio.info(file_path)

    if not (info.sample_rate == 44100 or info.sample_rate == 48000):
        print(f"{file_path} failed sample rate sanity check: {info.sample_rate}")
    if not (info.num_channels == 1 or info.num_channels == 2):
        print(f"{file_path} failed channel count sanity check: {info.num_channels}")

    return info.num_frames / info.sample_rate


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
    return val[::int(len(val) / 10)]

def main():
    train_duration = sum([check(f) for f in get_training_files()])
    print(f"train_duration: {train_duration}")

    validation_duration = sum([check(f) for f in get_validation_files()])
    print(f"validation_duration: {validation_duration}")

    demo_duration = sum([check(f) for f in get_demo_files()])
    print(f"demo_duration: {demo_duration}")


main()
