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

def compute_autoencoder_loss(expected, inference, hp):
    """
    expected: Tensor[batch_count x example_size]
    inference: Tensor[batch_count x example_size]
    hp: Hyperparameters

    returns: float. The loss of the autoencoder
    """

    loss = torch.nn.functional.mse_loss(inference, expected)
    return loss


class AutoEncoder(torch.nn.Module):
    def __init__(self, hp):
        """
        hp: Hyperparameters
        """
        super().__init__()

        self.hp = hp
        nonlinear = torch.nn.LeakyReLU()

        self.spec_out_time = 1 + int(hp.example_size / hp.hop_length)
        self.spec_out_height = 1 + (hp.n_fft // 2)
        self.d_model = int(math.ceil(self.spec_out_height / 8) * 8)

        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = hp.transformer_nhead,
            batch_first = True,
        )

        
        self.encoder = torch.nn.Sequential(
            torch.nn.TransformerEncoder(
                transformer_layer,
                num_layers = hp.transformer_num_layers
            ),

            torch.nn.Flatten(start_dim = 1, end_dim = -1),
            torch.nn.Linear(self.spec_out_time * self.d_model,
                            8000),
            nonlinear
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8000,
                            self.spec_out_time * self.d_model),
            nonlinear,
            torch.nn.Unflatten(1, (self.spec_out_time, self.d_model)),
            
            torch.nn.TransformerEncoder(
                transformer_layer,
                num_layers = hp.transformer_num_layers
            )
        )


    def forward(self, x):
        """
        x: Tensor[batch_count x unzipped_spectrogram_size]

        returns: Tensor[batch_count x unzipped_spectrogram_size]
        """

        # TransformerEncoder takes input with the shape [N x S x E],
        # where N is batch size, S is sequence length, E is embedding/feature dimension
        # but the spectrogram outputs [batch x freq x time]

        x = x.permute(0, 2, 1)
        # pad up x
        x = torch.nn.functional.pad(x, (0, self.d_model - self.spec_out_height),
                                    "constant", 0)

        x = self.encoder(x)
        x = self.decoder(x)

        # unpad x
        x = x[:, :, 0 : self.spec_out_height]

        x = x.permute(0, 2, 1)

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
    batches = make_batches(signal, hp)
    inference = model(batches)
    inference = reconstruct_waveform(inference, hp)
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
            print("starting file group...")
            file_group_end = file_group + file_group_size
            file_group_end = min(file_group_end, len(training_file_names))

            macrobatches = []
            print("cleared macrobatches...")
            for file_name in training_file_names[file_group : file_group_end]:
                signal, sample_rate = load_and_check(file_name, hp)
                print(f"loaded {file_name}")
                batches = make_batches(signal, hp)
                del signal
                print(f"made batches from {file_name}")

                chunk_indices = [hp.batch_size * i for i in range(
                    int(math.ceil(batches.shape[0] / hp.batch_size))
                )]
                for chunk_index in chunk_indices:
                    subscript_end = chunk_index + hp.batch_size
                    subscript_end = min(batches.shape[0], subscript_end)
                    macrobatches.append(batches[chunk_index:subscript_end])
                del batches
                del chunk_indices

            random.shuffle(macrobatches)

            print("shuffled macrobatches")

            previous_total_batches_from_epoch = total_batches_from_epoch

            # loop over batches in the file group...
            for batch in macrobatches:
                auto_encoder.train()
                optimizer.zero_grad()
                inference = auto_encoder(batch)
                loss = compute_autoencoder_loss(batch, inference, hp)

                loss.backward()
                optimizer.step()
                total_loss_from_epoch += float(loss)
                total_batches_from_epoch += int(batch.shape[0])

                del inference
                del batch
                del loss

                # print out current training stats, and validate occasionally
                training_stats_string = (
                    f"\r{epoch:02d}    "
                    f"| {total_batches_from_epoch}  "
                    f"| {total_loss_from_epoch / (total_batches_from_epoch + 1)}  "
                )

                print(training_stats_string + valid_stats_string, end="")

            del macrobatches
            should_validate = False
            if epoch > 0 or total_batches_from_epoch > 0:
                a = total_batches_from_epoch / hp.validate_every_n_batches
                b = previous_total_batches_from_epoch / hp.validate_every_n_batches

                if int(a) > int(b):
                    should_validate = True

            if should_validate:
                print("validating...")
                with torch.no_grad():
                    auto_encoder.eval()
                    valid_loss = 0
                    valid_denom = 0
                    for valid_file in validation_file_names:
                        signal, sample_rate = load_and_check(valid_file, hp)
                        batches = make_batches(signal, hp)
                        inference = auto_encoder(batches)
                        valid_loss += float(compute_autoencoder_loss(expected,
                                                                     inference,
                                                                     hp))
                        valid_denom += int(batches.shape[0])

                        if valid_denom != 0:
                            valid_stats_string = f"| {valid_loss / valid_denom}"
                            print(training_stats_string + valid_stats_string, end="")

                        del signal
                        del batches
                        del inference


        print("")
        if on_finish_epoch:
            on_finish_epoch(epoch)

    return auto_encoder


def main():
    hp = Hyperparameters()
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



