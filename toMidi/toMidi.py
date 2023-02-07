from .toMidi_data import *
import subprocess
import os
import config.config

def main():
    print("toMidi")

def make_pitch_interval_files():
    files = get_training_files() + get_validation_files() + get_test_files()
    files = [os.path.splitext(f)[0] + ".midi" for f in files]

    helper_script = config.config._get_config_file_path()
    helper_script = os.path.dirname(helper_script)
    helper_script = os.path.join(helper_script, "atrom-repo/misc/midiToPitchInterval.py")

    for i, file in enumerate(files):
        args = []
        args += ["python"]
        args += [helper_script]
        args += [file]
        args += ["-y"]

        subprocess.run(args)
        print(f"{i + 1} of {len(files) + 1}")

    print("Finished")
