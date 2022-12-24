import os
import json
import inspect
import socket
import pprint

def _get_config_file_path() -> str:
    """
    returns: The file path for the config file
    """
    
    import inspect
    # some_dir/atrom-repo/config/config.py
    source_file = inspect.getsourcefile(_get_config_file_path)
    # some_dir/atrom-repo/config
    source_file = os.path.dirname(source_file)
    # some_dir/atrom-repo
    source_file = os.path.dirname(source_file)
    # some_dir
    source_file = os.path.dirname(source_file)

    # some_dir/config.txt
    return os.path.join(source_file, "config.txt")


def _get_user_confirmation(prompt: str, suffix: str=" (y/n) ") -> bool:
    """
    Prompts the user at stdin
    prompt: The prompt to display at sdout
    suffix: The suffix to append at the end of the prompt

    returns: bool
    """
    response = ""
    while response != "y" and response != "n":
        response = input(prompt + suffix).lower()

    return response == "y"


def _recursively_find_files_in_dir(dir):
    """
    dir: string. The root directory to search at
    
    returns: list[string]. Each path starts with `dir`
    """

    result = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for file in filenames:
            result.append(os.path.join(dirpath, file))
    return result


def _is_wav_file(filename):
    """
    filename: string

    returns: bool. True if the file has a .wav extension, and False otherwise
    """
    return os.path.splitext(filename)[1] == ".wav"


def _write_data_file(config_json_path: list[str], lines: list[str]):
    """
    config_json_path: the keys to determine the type of data file
    lines: the data (joined by newlines) to write to disk
    """
    with open(read_config_value(config_json_path), "w") as f:
        f.write("\n".join(lines))


def _make_autoencoder_train_file():
    if socket.gethostname() == "mithril.local":
        # maddy's local machine
        search_dir = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio"
        files = ["lig_orchestra.wav", "lig_vocals.wav", "myshot.wav"]
        files = [os.path.join(search_dir, f) for f in files]

    elif socket.gethostname() == "memstar":
        search_dir = "/z/atrom/datasets/unabeled/YouTube"
        files = [f for f in _recursively_find_files_in_dir(search_dir)
                 if _is_wav_file(f)]
        files = files[:int(len(files) * 7 / 10)]

    _write_data_file(["autoencoder", "train"], files)


def _make_autoencoder_validation_file():
    if socket.gethostname() == "mithril.local":
        # maddy's local machine
        search_dir = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio"
        files = ["lig_soundtrack.wav", "all-star.wav"]
        files = [os.path.join(search_dir, f) for f in files]

    elif socket.gethostname() == "memstar":
        search_dir = "/z/atrom/datasets/unabeled/YouTube"
        files = [f for f in _recursively_find_files_in_dir(search_dir)
                 if _is_wav_file(f)]
        files = files[int(len(files) * 7 / 10) :
                      int(len(files) * 85 / 100)]

    _write_data_file(["autoencoder", "validation"], files)


def _make_autoencoder_demo_file():
    if socket.gethostname() == "mithril.local":
        # maddy's local machine
        search_dir = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio"
        files = ["lig_orchestra.wav", "lig_vocals.wav", "myshot.wav",
                 "lig_soundtrack.wav", "all-star.wav"]
        files = [os.path.join(search_dir, f) for f in files]

    elif socket.gethostname() == "memstar":
        search_dir = "/z/atrom/datasets/unabeled/YouTube"
        files = [f for f in _recursively_find_files_in_dir(search_dir)
                 if _is_wav_file(f)]
        files = files[int(len(files) * 7 / 10) :
                      int(len(files) * 85 / 100)]
        files = files[::int(len(files) / 20)]

    _write_data_file(["autoencoder", "demo"], files)

    
def _load_config_file() -> dict:
    """
    Loads the contents of the config file as json

    returns: The json representation of the contents of the config file
    """
    with open(_get_config_file_path()) as f:
        return json.loads(f.read())


    
def make_config_file():
    """
    Makes/overwrites the config file. Prompts the user for confirmation at stdin
    """
    
    config_path = _get_config_file_path()

    if not _get_user_confirmation(f"Create config file at {config_path}?"):
        print("Exiting")
        return

    autoencoder_train_file = os.path.join(os.path.dirname(config_path),
                                          "autoencoder_train_file.txt")
    autoencoder_validation_file = os.path.join(os.path.dirname(config_path),
                                               "autoencoder_validation_file.txt")
    autoencoder_demo_file = os.path.join(os.path.dirname(config_path),
                                         "autoencoder_demo_file.txt")
    autoencoder_checkpoint_file = os.path.join(os.path.dirname(config_path),
                                               "autoencoder_checkpoint_file")
    autoencoder_demo_write_dir = os.path.join(os.path.dirname(config_path),
                                              "autoencoder_demo_write")

    config_data = {
        "autoencoder" : {
            "train" : autoencoder_train_file,
            "validation" : autoencoder_validation_file,
            "demo" : autoencoder_demo_file,
            "checkpoint" : autoencoder_checkpoint_file,
            "demo_write" : autoencoder_demo_write_dir
        }
    }


    print("Config file contents:")
    pprint.pprint(config_data)

    if not _get_user_confirmation("Create config file?"):
        print("Exiting")
        return

    with open(config_path, "w") as f:
        f.write(json.dumps(config_data))


def hacky_write_data_files():
    # hacky way of writing data files, should be improved
    _make_autoencoder_train_file()
    _make_autoencoder_validation_file()
    _make_autoencoder_demo_file()


def read_data_file(config_json_path: list[str]) -> list[str]:
    """
    config_json_path: the keys to determine the type of data file
    
    returns: the contents (split by newlines) read from disk
    """
    with open(read_config_value(config_json_path)) as f:
        return f.read().splitlines()

def read_config_value(config_json_path: list[str]) -> str:
    """
    config_json_path: the keys to the value
    
    returns: the value at the end of the path
    """
    j = _load_config_file()
    for component in config_json_path:
        j = j[component]

    return j
