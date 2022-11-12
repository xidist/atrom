import torch
import torchaudio
import math
import random

torch.manual_seed(0)
random.seed(0)
print("finished importing modules...")

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
    assert info.sample_rate == 44100 or info.sample_rate == 48000
    assert info.num_channels == 1 or info.num_channels == 2
    
    waveform, sample_rate = torchaudio.load(file_path)

    if info.num_channels == 2:
        waveform = torch.sum(waveform, dim=0)

    # squeeze out the channel dimension, since it's already mono
    waveform = waveform.squeeze()
    waveform = waveform.to(hp.device)

    if info.sample_rate != hp.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, hp.sample_rate)
    
    return waveform, sample_rate


def make_batches(signal, hp):
    """
    signal: Tensor[frame_length] containing the audio data
    hp: Hyperparameters
    returns: Tensor[batch_size x (left_context + source + right_context)].
             A tensor of batches, where each batch is a contiguous slice of audio data. 
             May be left or right padded with zeros at its ends
    """
    source_start_times = torch.arange(start=0, end=signal.shape[0], step=hp.batch_offset, dtype=int)
    
    signal = torch.nn.functional.pad(signal, (hp.left_context, hp.source + hp.right_context),
                                     "constant", 0)
    subscript_length = hp.left_context + hp.source + hp.right_context

    # make a [batch x subscript_length] tensor for indexing signal
    indices = torch.arange(start=0, end=subscript_length, step=1, dtype=int)
    indices = indices.unsqueeze(0).repeat(source_start_times.shape[0], 1)
    # make a [batch x subscript_length] tensor for shifting indices
    offset = source_start_times.unsqueeze(1).repeat(1, subscript_length)
    # add source_start_times[i] to each indices[i]
    indices += offset

    return signal[indices]

def make_expected_from_batches(batches, hp):
    """
    batches: Tensor[batch_count x left_context + source + right_context]. Audio signals
             for the autoencoder to reproduce
    hp: Hyperparameters

    returns: Tensor[batch_count x source]. The center slice of each batch we want
             the autoencoder to reproduce
    """
    # make a [source] tensor for subscripting batches
    indices = torch.arange(start=0, end=hp.source, step=1, dtype=int) + hp.left_context

    result =  batches[:, indices]
    return result

def put_inference_inside_batches(batches, inference, hp):
    """
    batches: Tensor[batch_count x left_context + source + right_context]. Audio signals
             for the autoencoder to reproduce
    inference: Tensor[batch_count x source]. The audio signal the autoencoder actually reproduced
    hp: Hyperparameters

    returns: Tensor[batch_count x left_context + source + right_context]. A copy of batches,
             with the source samples replaced with the contents of inference
    """
    b_copy = batches.clone()
    # make a [source] tensor for subscripting b_copy
    indices = torch.arange(start=0, end=hp.source, step=1, dtype=int) + hp.left_context
    b_copy[:, indices] = inference
    return b_copy

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
            nonlinear,
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

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Hyperparameters:
    """
    Bag of hyperparameters that need to be passed around at times
    """
    def __init__(self,
                 sample_rate = 16000,
                 left_context = 16000,
                 source = 16000,
                 right_context = 16000,
                 batch_offset = 16000,
                 batch_size = 8,
                 learning_rate = 1e-4,
                 weight_decay = 1e-5,
                 epochs = 10,
                 validate_every_n = 10,
                 e2e_every_n = 1,
                 n_mels = 128,
                 hop_length = 2000,
                 n_fft = 4000):

        self.sample_rate = sample_rate
        self.left_context = left_context
        self.source = source
        self.right_context = right_context
        self.batch_offset = batch_offset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = 10
        self.validate_every_n = 10
        self.e2e_every_n = e2e_every_n
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using CUDA acceleration!")

    def make_spectrogram_object(self):
        """
        Makes a MelSpectrogram module
        """
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            normalized=True).to(self.device)

def end_to_end(model, hp, source_file, dest_file):
    """
    Runs the contents of `source_file` through the AutoEncoder `model`, writing the 
    reconstructed audio file into `dest_file`. 

    model: AutoEncoder
    hp: Hyperparameters
    source_file: str
    dest_file: str
    """
    signal, sample_rate = load_and_check(source_file, hp)
    spec_obj = hp.make_spectrogram_object()
    batches = make_batches(signal, hp)
    inference = model(batches, spec_obj)
    inference = inference.reshape(-1)
    inference = inference.unsqueeze(0)

    out_sample_rate = 44100
    inference = torchaudio.functional.resample(inference, hp.sample_rate, out_sample_rate)
    inference = inference.cpu()
    torchaudio.save(dest_file, inference, out_sample_rate)


def train_model(hp, auto_encoder, training_file_names, validation_file_names,
                starting_epoch=0, on_finish_epoch=None):
    """
    The basic idea is that we loop over some unlabeled training examples, and for each file,
    compute the loss and update the autoencoder. Since the autoencoder can only focus on
    a few seconds of audio at a time, we split the entire song (which might be many minutes)
    into batches that cover its duration

    hp: Hyperparameters
    auto_encoder: AutoEncoder
    training_file_names: list[string]
    validation_file_names: list[string]
    starting_epoch: int = 0
    on_finish_epoch: (int) -> void = None
    """


    optimizer = torch.optim.AdamW(auto_encoder.parameters(),
                                  lr=hp.learning_rate,
                                  weight_decay=hp.weight_decay)

    print("Epoch | Batch | Train Loss | Valid Loss")

    valid_stats_string = ""

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
            for batch in macrobatches:
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

            if epoch > 0 and epoch % hp.validate_every_n == 0:
                with torch.no_grad():
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


def maddy_local_testing():
    root_dir = "../foobar-maddy/"
    input_audio_dir = root_dir + "input_audio/"
    output_audio_dir = root_dir + "output_audio/"
    models_dir = root_dir + "models/"
    
    training = [input_audio_dir + "myshot.wav",
                input_audio_dir + "lig_orchestra.wav",
                input_audio_dir + "lig_vocals.wav",
                ]
    validation = [input_audio_dir + "lig_soundtrack.wav",
                  input_audio_dir + "all-star.wav"]
    
    hp = Hyperparameters()
    hp.batch_size = 32
    hp.epochs = 20
    hp.learning_rate = 1e-5
    hp.left_context = 0
    hp.right_context = 0
    auto_encoder = AutoEncoder(hp)

    starting_epoch = 0
    if starting_epoch != 0:
        auto_encoder.load(models_dir + f"e{starting_epoch - 1}")
    auto_encoder = auto_encoder.to(hp.device)

    print("finished creating auto_encoder...")

    def on_finish_epoch(n):
        import os
        os.mkdir(output_audio_dir + f"e{n}")

        end_to_end_files = [
            "lig_soundtrack.wav",
            "lig_orchestra.wav",
            "lig_vocals.wav",
            "myshot.wav",
            "all-star.wav",
        ]

        last_epoch = n == starting_epoch + hp.epochs - 1

        if last_epoch or n % hp.e2e_every_n == 0:
            for file_name in end_to_end_files:
                end_to_end(auto_encoder, hp,
                           source_file=input_audio_dir + file_name,
                           dest_file=output_audio_dir + f"e{n}/" + file_name)

        if last_epoch or n % 3 == 0:
            auto_encoder.save(models_dir + f"e{n}")

    train_model(hp, auto_encoder, training, validation,
                starting_epoch=starting_epoch, on_finish_epoch=on_finish_epoch)


maddy_local_testing()
