import torch
import torchaudio
import math
import random
import os
from .autoencoder_util import *
from .autoencoder_data import *
from config.config import *

torch.manual_seed(0)
random.seed(0)
print("finished importing modules...")


def compute_autoencoder_loss(batches, expected, inference, spec_objs, hp):
    """
    batches: Tensor[batch_count x left_context + source + right_context]. Raw audio signal
    expected: Tensor[batch_count x source]
    inference: Tensor[batch_count x source]
    spec_objs: A list of objects capable of computing a spectrogram. 
               The MSE will be computed using each spectrogram object, comparing the 
               spectrogram between using batches, and the spectrogram putting inference
               in the middle of batches
    hp: Hyperparameters

    returns: float. The loss of the autoencoder
    """

    # note: if inference is drawn from a normal distribution,
    # loss from wave-to-wave comparison is 1,
    # and loss from spectrogram comparison is 158.
    # do we need to find a way to balance these to the same order of magnitude?

    loss = torch.nn.functional.mse_loss(inference, expected)
    batches_with_inference = put_inference_inside_batches(batches, inference, hp)
    
    for spec_obj in spec_objs:
        inf_spec = spec_obj(batches_with_inference)
        exp_spec = spec_obj(batches)
        loss += torch.nn.functional.mse_loss(inf_spec, exp_spec) / 10000

    # mse averages over the batches, but we don't want to
    loss *= batches.shape[0]

    return loss


class AutoEncoder(torch.nn.Module):
    def __init__(self, hp):
        """
        hp: Hyperparameters
        """
        super().__init__()

        self.hp = hp
        nonlinear = torch.nn.LeakyReLU()

        in_size = hp.left_context + hp.source + hp.right_context
        out_size = hp.source
        self.spec_size = hp.n_mels * (1 + math.floor(in_size / hp.hop_length))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_size + self.spec_size, 8000),
            nonlinear
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8000, out_size)
        )


    def forward(self, x, spec_obj):
        """
        x: Tensor[batch_count x left_context + source + right_context]
        spec_obj: An object to use to first compute the spectrogram

        returns: Tensor[batch_count x source]
        """

        s = spec_obj(x)
        s = s.reshape(s.shape[0], -1)
        x = torch.cat((x, s), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)

        return x



def end_to_end(model, hp, source_file, dest_file):
    """
    Runs the contents of `source_file` through the AutoEncoder `model`, writing the 
    reconstructed audio file into `dest_file`. 

    model: AutoEncoder
    hp: Hyperparameters
    source_file: str
    dest_file: str
    """
    model.eval()
    signal, sample_rate = load_and_check(source_file, hp)
    spec_obj = hp.make_spectrogram_object()
    batches = make_batches(signal, hp)
    inference = model(batches, spec_obj)
    inference = inference.reshape(-1)
    inference = inference.unsqueeze(0)

    out_sample_rate = 44100
    inference = torchaudio.functional.resample(inference, hp.sample_rate, out_sample_rate)
    inference = inference.cpu()
    if not os.path.exists(os.path.dirname(dest_file)):
        os.makedirs(os.path.dirname(dest_file))
    torchaudio.save(dest_file, inference, out_sample_rate)


def train_model(hp, auto_encoder, optimizer,
                training_file_names, validation_file_names,
                starting_epoch=0, on_finish_epoch=None):
    """
    The basic idea is that we loop over some unlabeled training examples, and for each file,
    compute the loss and update the autoencoder. Since the autoencoder can only focus on
    a few seconds of audio at a time, we split the entire song (which might be many minutes)
    into batches that cover its duration

    hp: Hyperparameters
    auto_encoder: AutoEncoder
    optimizer: A torch optimizer (e.g. AdamW, SDG)
    training_file_names: list[string]
    validation_file_names: list[string]
    starting_epoch: int = 0
    on_finish_epoch: (int) -> void = None
    """

    print("Epoch | Batch | Train Loss | Valid Loss")

    valid_stats_string = ""

    # loop over epochs...
    for epoch in range(starting_epoch, starting_epoch + hp.epochs):
        random.shuffle(training_file_names)
        total_loss_from_epoch = 0
        total_batches_from_epoch = 0
        training_stats_string = ""


        file_group_size = 16
        file_groups = [file_group_size * i for i in range(
            int(math.ceil(len(training_file_names) / file_group_size))
        )]
        random.shuffle(file_groups)

        # loop over groups of training files...
        for file_group in file_groups:
            file_group_end = file_group + file_group_size
            file_group_end = min(file_group_end, len(training_file_names))

            macrobatches = []
            for file_name in training_file_names[file_group : file_group_end]:
                signal, sample_rate = load_and_check(file_name, hp)
                batches = make_batches(signal, hp)
                expected = make_expected_from_batches(batches, hp)

                chunk_indices = [hp.batch_size * i for i in range(
                    int(math.ceil(batches.shape[0] / hp.batch_size))
                )]
                for chunk_index in chunk_indices:
                    subscript_end = chunk_index + hp.batch_size
                    subscript_end = min(batches.shape[0], subscript_end)
                    macrobatches.append((batches[chunk_index:subscript_end],
                                         expected[chunk_index:subscript_end]))


            random.shuffle(macrobatches)
            spec_obj = hp.make_spectrogram_object()

            previous_total_batches_from_epoch = total_batches_from_epoch

            # loop over batches in the file group...
            for batch in macrobatches:
                auto_encoder.train()
                optimizer.zero_grad()
                x = batch[0]
                y = batch[1]
                inference = auto_encoder(x, spec_obj)
                spec_objs = []
                loss = compute_autoencoder_loss(x, y, inference, spec_objs, hp)

                loss.backward()
                optimizer.step()
                total_loss_from_epoch += loss
                total_batches_from_epoch += x.shape[0]

                # print out current training stats, and validate occasionally
                training_stats_string = (
                    f"\r{epoch:02d}    "
                    f"| {total_batches_from_epoch}  "
                    f"| {total_loss_from_epoch / (total_batches_from_epoch + 1)}  "
                )

                print(training_stats_string + valid_stats_string, end="")

            should_validate = False
            if epoch > 0:
                a = total_batches_from_epoch / hp.validate_every_n_batches
                b = previous_total_batches_from_epoch / hp.validate_every_n_batches

                if int(a) > int(b):
                    should_validate = True

            if should_validate:
                with torch.no_grad():
                    auto_encoder.eval()
                    valid_loss = 0
                    valid_denom = 0
                    for valid_file in validation_file_names:
                        signal, sample_rate = load_and_check(valid_file, hp)
                        batches = make_batches(signal, hp)
                        expected = make_expected_from_batches(batches, hp)
                        inference = auto_encoder(batches, spec_obj)
                        spec_objs = []
                        valid_loss += compute_autoencoder_loss(batches,
                                                               expected,
                                                               inference,
                                                               spec_objs, hp)
                        valid_denom += batches.shape[0]

                        if valid_denom != 0:
                            valid_stats_string = f"| {valid_loss / valid_denom}"
                            print(training_stats_string + valid_stats_string, end="")


        print("")
        if on_finish_epoch:
            on_finish_epoch(epoch)

    return auto_encoder


def main():
    hp = Hyperparameters()
    hp.left_context = 0
    hp.right_context = 0
    hp.batch_size = 32
    hp.learning_rate = 1e-5
    hp.epochs = 20
    hp.validate_every_n_batches = 3600 * 10
    auto_encoder = AutoEncoder(hp)

    optimizer = torch.optim.AdamW(auto_encoder.parameters(),
                                  lr=hp.learning_rate,
                                  weight_decay=hp.weight_decay)


    def save_training(epoch):
        """
        epoch: int. The most recently completed epoch
        """
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : auto_encoder.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()
        }, get_checkpoint_file_path())

    def load_training(model, optimizer):
        """
        model: An already initizlied AutoEncoder. (The weights will be set
               using the data loaded from disk, but the model structure must
               be the same)
        optimizer: The torch optimizer used for training. Must be the same type
                   when saving and loading
        
        returns: int. The most recently completed epoch
        """
        if os.path.exists(get_checkpoint_file_path()):
            checkpoint = torch.load(get_checkpoint_file_path())
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return checkpoint["epoch"] + 1

        else:
            return 0

    starting_epoch = load_training(auto_encoder, optimizer)
    auto_encoder = auto_encoder.to(hp.device)

    print("finished creating auto_encoder...")

    def on_finish_epoch(n):
        this_epoch_demo_directory = os.path.join(get_demo_write_directory(), f"e{n}")

        last_epoch = n == starting_epoch + hp.epochs - 1

        if last_epoch or n % hp.demo_every_n_epochs == 0:
            for path in get_demo_files():
                last_path_component = os.path.basename(os.path.normpath(path))
                dest_file = os.path.join(this_epoch_demo_directory, last_path_component)
                end_to_end(auto_encoder, hp,
                           source_file=path,
                           dest_file=dest_file)

        save_training(n)


    
    train_model(hp, auto_encoder, optimizer,
                get_training_files(), get_validation_files(),
                starting_epoch=starting_epoch, on_finish_epoch=on_finish_epoch)


