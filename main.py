import os
import socket
import pprint
import json
import inspect
import config.config
import argparse
import autoencoder.autoencoder
import torchaudio
import subprocess
import matplotlib.pyplot as plt
import toMidi.toMidi

class App:
    parser = argparse.ArgumentParser()
    commands = []

    @staticmethod
    def run():
        args = App.parser.parse_args()

        for command in App.commands:
            if args.__dict__[command.__name__]:
                command()
                exit()

        App.parser.print_help()

def cli_accessible(f):
    App.parser.add_argument(f"--{f.__name__}", action="store_true")
    App.commands.append(f)
    return f


def _check_no_load(file_path, verbose=True):
    """
    Performs the same checks as load_and_check from autoencoder.py

    file_path: str, the path to the audio file to check.

    returns: (float, bool). the duration of the file, in seconds.
                            True if the file loaded without errors, else False
    """
    info = torchaudio.info(file_path)

    no_error = True

    if not (info.sample_rate == 44100 or info.sample_rate == 48000):
        no_error = False
        if verbose:
            print(f"{file_path} failed sample rate sanity check: {info.sample_rate}")
    if not (info.num_channels == 1 or info.num_channels == 2):
        no_error = False
        if verbose:
            print(f"{file_path} failed channel count sanity check: {info.num_channels}")

    return (info.num_frames / info.sample_rate, no_error)


@cli_accessible
def make_config_file():
    config.config.make_config_file()

@cli_accessible    
def hacky_write_data_files():
    config.config.hacky_write_data_files()


@cli_accessible    
def find_dataset_errors(verbose=True):
    """
    Prints out any files that fail to load properly

    returns: list[string]. The files that failed to load
    """
    errors = []
    errors += [f for f in
               config.config.read_data_file(["autoencoder", "train"])
               if not _check_no_load(f, verbose=False)[1]]
    errors += [f for f in
               config.config.read_data_file(["autoencoder", "validation"])
               if not _check_no_load(f, verbose=False)[1]]

    if verbose:
        if errors:
            for f in errors:
                print(f)
            print(f"\n{len(errors)} mistakes found")
        else:
            print("No errors")
    return errors



def _fix_dataset_file(input_file):
    """
    Runs ffmpeg to convert the input file to mono 16 bit 44.1 kHz audio.
    Modifies the original file
    """

    """
    Run ffmpeg,
    automatically answering "yes" if asked if we want to overwrite any output files,
    using the given input file,
    setting the number of audio channels to 1,
    setting the audio rate to 44100 Hz,
    setting the sample format to signed 16 bit integer,
    to the given output file. 

    Then, move the temp file on top of the input file
    """

    temp_file = "/tmp/fix-wave-format.wav"

    args = []
    args += ["ffmpeg"]
    args += ["-y"]
    args += ["-i", input_file]
    args += ["-ac", "1"]
    args += ["-ar", "44100"]
    args += ["-sample_fmt", "s16"]
    args += [temp_file]

    print(" ".join(args))
    subprocess.run(args)

    args = []
    args += ["mv"]
    args += [temp_file]
    args += [input_file]

    print(" ".join(args))
    subprocess.run(args)


@cli_accessible
def fix_dataset_errors():
    for f in find_dataset_errors(verbose=False):
        _fix_dataset_file(f)

        
@cli_accessible    
def print_dataset_stats():
    """
    Prints out stats about the datasets
    """
    
    train_duration = sum([_check_no_load(f)[0] for f in
                          config.config.read_data_file(["autoencoder", "train"])])
    print(f"train_duration: {train_duration / 3600} hours")
    
    validation_duration = sum([_check_no_load(f)[0] for f in
                               config.config.read_data_file(["autoencoder", "validation"])])
    print(f"validation_duration: {validation_duration / 3600} hours")

    demo_duration = sum([_check_no_load(f)[0] for f in
                         config.config.read_data_file(["autoencoder", "demo"])])
    print(f"demo_duration: {demo_duration / 3600} hours")


@cli_accessible
def train_autoencoder():
    autoencoder.autoencoder.main()

    
@cli_accessible
def summarize_log_file(logFile="nohup.out", everyN=100, alwaysPrintLast=True):
    digitCtr = 0
    with open(logFile) as f:
        lines = f.readlines()
        for l in lines:
            if l[0].isdigit():
                digitCtr += 1

            if not l[0].isdigit() or (digitCtr % everyN == 0):
                print(l, end="")

        if alwaysPrintLast:
            print(lines[-1])

@cli_accessible
def graph_loss(logFile="nohup.out", everyN=1):
    digitCtr = 0
    batches = []
    train_losses = []
    val_losses = []
    maxBatchN = -1
    maxEpochN = 0
    endOfEpochs = []

    with open(logFile) as f:
        lines = f.readlines()

    # figure out the number of batches in an epoch
    for l in lines:
        if l[0].isdigit():
            batch = int(l.split("|")[1].strip())
            if batch >= maxBatchN:
                maxBatchN = batch
            else:
                break
    batchesPerEpoch = maxBatchN + 1


    # figure out the number of epochs
    for l in lines:
        if l[0].isdigit():
            maxEpochN = max(maxEpochN, int(l.split("|")[0].strip()))
    nEpochs = maxEpochN + 1


    # build the training losses,
    # the validation losses,
    # and the end of epoch losses
    for l in lines:
        if l[0].isdigit():
            digitCtr += 1

        if l[0].isdigit() and (digitCtr % everyN == 0):
            epoch = int(l.split("|")[0].strip())
            batch = int(l.split("|")[1].strip()) / 3600
            train_loss = float(l.split("|")[2].strip().split(" ")[0])
            batches.append((epoch * batchesPerEpoch / 3600) + batch)
            train_losses.append(train_loss)

            if len(l.split("|")) > 3:
                # todo: remove the starting/validating hard coded strings
                # in parsing here, once fixing autoencoder.py
                val_loss = float(l.split("|")[3].strip().split("starting")[0].split("validating")[0])
                val_losses.append(((epoch * batchesPerEpoch / 3600) + batch, val_loss))
                if len(val_losses) >= 2:
                    if val_losses[-2][1] == val_losses[-1][1]:
                        del val_losses[-1]
                    elif val_losses[-2][0] == val_losses[-1][0]:
                        del val_losses[-2]
                    

            if epoch >= len(endOfEpochs):
                endOfEpochs.append((epoch, batch, train_loss))
            else:
                if endOfEpochs[-1][1] < batch:
                    endOfEpochs[-1] = ((epoch, batch, train_loss))

    plt.plot(batches, train_losses)

    plt.plot([(e * batchesPerEpoch / 3600) + b for e, b, _ in endOfEpochs],
             [l for _, _, l in endOfEpochs],
             color="red")

    for i in range(nEpochs):
        plt.axvline(i * batchesPerEpoch / 3600, color="black")

    plt.plot([x[0] for x in val_losses], [x[1] for x in val_losses], color="green")

    print("training losses at end of epochs:")
    for e, b, l in endOfEpochs:
        print(f"{e} ({b}): {l}")

    for x in val_losses:
        print(x)
    
    plt.show()

@cli_accessible
def toMidi_make_pitch_interval_files():
    toMidi.toMidi.make_pitch_interval_files()

@cli_accessible
def train_toMidi():
    toMidi.toMidi.train_toMidi()
    
if __name__ == "__main__":
    App.run()
