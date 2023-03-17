from .toMidi_data import *
from .toMidi_eval import *
import subprocess
import os
from config.config import *
import torch
import torchaudio
import math
import random


torch.manual_seed(0)
random.seed(0)
print("finished importing modules...")


def make_pitch_interval_files():
    files = get_training_files() + get_validation_files() + get_test_files()
    files = [os.path.splitext(f)[0] + ".midi" for f in files]

    helper_script = _get_config_file_path()
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


class Hyperparameters:
    def __init__(self,
                 clipLength: float=1,
                 maxPredictedTokens: int=100):
        self.clipLength = clipLength
        self.maxPredictedTokens = maxPredictedTokens

class ToMidi(torch.nn.Module):
    def __init__(self, hp: Hyperparameters)
        super().__init__()
        self.hp = hp

def train_model(model: ToMidi, optimizer,
                starting_epoch=0, on_finish_epoch=None):
    for epoch in range(starting_epoch, starting_epoch + hp.epochs):
        total_loss_from_epoch = 0
        total_batches_from_epoch = 0
        valid_stats_string = ""
        
        raise Exception("get training files")
        training_data = [] # [(wavBuffer, [numberTokens])]
        random.shuffle(training_data)
        for example_index, example in enumerate(training_data):
            print(f"starting example {example_index + 1} of {len(training_data)}")

            raise Exception("chopExampleIntoClips")
            clips = chopExampleIntoClips(example) # ([clipBuffer], [clipTokens])
            model.train()
            optimizer.zero_grad()
            embeded_clip_tokens = model.embedOutputTokens(clips[1])

            raise Exception("teacher forcing somehow")
            loss.backward()
            optimizer.step()
            total_loss_from_epoch += float(loss)
            total_batches_from_epoch += int(len(clips[0]))
            
            training_stats_string = (
                f"\r{epoch:02d}    "
                f"|  {total_batches_from_epoch}  "
                f"|  {total_loss_from_epoch / (total_batches_from_epoch + 1)}  "
            )

            print(training_stats_string + valid_stats_string, end="")

            raise Exception("validate occasionally")
            if should_validate:
                print("validating...")
                metrics = validate_model(model)
                raise Exception("process metrics")

                if valid_denom != 0:
                    valid_stats_string = f"| {valid_loss / valid_denom}"
                    print(training_stats_string + valid_stats_string, end="")

def validate_model(model: ToMidi):
    raise Exception("get validation files")
    validation_data = [] # [(wavBuffer, [numberTokens])]
    predictions = predict(model, [example[0] for example in validation_data])
    raise Exception("evaluate_metrics")
    metrics = evaluate_metrics(predictions, [example[1] for example in validation_data])
    return metrics

def predict(model: ToMidi, to_predict):
    """
    to_predict: [wavBuffer], where wavBuffer = [float]
    """
    result = []
    for item_index, item in enumerate(to_predict):
        print(f"starting prediction {item_index + 1} of {len(to_predict)}")

        raise Exception("chopItemIntoClips")
        clips = chopItemIntoClips(item) # [clipBuffer]
        with torch.no_grad():
            model.eval()
            predictions = []
            for clip in clips:
                raise Exception("detect eos")
                while len(clipPredictions) < model.hp.maxPredictedTokens and (len(clipPredictions) == 0 or clipPredictions[-1] == EOS):
                    clipPredictions.append(model.predictNext(clip, clipPredictions))
                predictions.append(clipPredictions)
            result.append(predictions)
    return result

        
"""
Training on one song: 
    chop it up into clips
    for each clip:
        wav2vec it
        embed the output tokens
        train the transformer via teacher forcing

Transcribing one song:
    chop it up into clips
    for each clip:
        wav2vec it
        until EOS is predicted or max tokens predicted
            embed the predicted tokens
            autoregressive sample the transformer for the next token
    stitch clip predictions together
"""


def main():
    print("toMidi")
