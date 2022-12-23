import os
import json
import inspect

def get_user_confirmation(prompt: str, suffix: str=" (y/n) ") -> bool:
    response = ""
    while response != "y" and response != "n":
        response = input(prompt + suffix).lower()

    return response == "y"

def get_config_file_path() -> str:
    import inspect
    # some_dir/atrom-repo/config/config.py
    source_file = inspect.getsourcefile(get_config_file_path)
    # some_dir/atrom-repo/config
    source_file = os.path.dirname(source_file)
    # some_dir/atrom-repo
    source_file = os.path.dirname(source_file)
    # some_dir
    source_file = os.path.dirname(source_file)

    # some_dir/config.txt
    return os.path.join(source_file, "config.txt")

def make_config_file():
    config_path = get_config_file_path()

    if not get_user_confirmation(f"Create config file at {config_path}?"):
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

    if not get_user_confirmation("Create config file?"):
        print("Exiting")
        return

    with open(config_path, "w") as f:
        f.write(json.dumps(config_data))

def load_config_file():
    with open(get_config_file_path()) as f:
        return json.loads(f.read())
