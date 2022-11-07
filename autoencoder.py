import torch
import torchaudio
import math
import random
print("finished importing modules...")

def load_and_check(file_path):
    """
    file_path: str, the path to the audio file to load
    returns: a tuple of (signal, sample_rate)
    signal: a Tensor[sample_length] containing the audio data, in the range -1 to 1
    sample_rate: int, the number of samples recorded each second
    """

    # lets do some basic sanity checks on the file format
    # these should hold on maestro, but might be too strict for youtube/JMS?
    # if so, we'll have to think about how to handle the failures
    info = torchaudio.info(file_path)
    assert info.sample_rate == 44100
    assert info.num_channels == 1
    
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.squeeze()
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
        # todo: magic number 158
        loss += torch.nn.functional.mse_loss(inf_spec, exp_spec) / 158

    return loss

class AutoEncoder(torch.nn.Module):
    def __init__(self, hp):
        """
        hp: Hyperparameters
        """
        super().__init__()


        spec_length = (hp.left_context + hp.source + hp.right_context) / hp.hop_length
        spec_length = int(math.ceil(spec_length))

        nonlinear = torch.nn.LeakyReLU()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hp.n_mels * spec_length, 4096),
            nonlinear,
            torch.nn.Linear(4096, 2048),
            nonlinear,
            torch.nn.Linear(2048, 512),
            nonlinear,
            torch.nn.Linear(512, 256)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            nonlinear,
            torch.nn.Linear(1024, 4096),
            nonlinear,
            torch.nn.Linear(4096, 16384),
            nonlinear,
            torch.nn.Linear(16384, 32768),
            nonlinear,
            torch.nn.Linear(32768, hp.source)
        )

    def forward(self, x, spec_obj):
        """
        x: Tensor[batch_count x left_context + source + right_context]
        spec_obj: An object to use to first compute the spectrogram

        returns: Tensor[batch_count x source]
        """


        speced = spec_obj(x)
        speced = torch.reshape(speced, (speced.shape[0], -1))
        encoded = self.encoder(speced)
        decoded = self.decoder(encoded)

        return decoded

class Hyperparameters:
    """
    Bag of hyperparameters that need to be passed around at times
    """
    def __init__(self,
                 left_context = 44100,
                 source = 44100,
                 right_context = 44100,
                 batch_offset = 44100,
                 learning_rate = 1e-4,
                 weight_decay = 1e-5,
                 epochs = 10,
                 validate_every_n = 10,
                 n_mels = 128,
                 hop_length = 2048):
        
        self.left_context = left_context
        self.source = source
        self.right_context = right_context
        self.batch_offset = batch_offset
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = 10
        self.validate_every_n = 10
        self.n_mels = n_mels
        self.hop_length = hop_length

    def make_spectrogram_object(self, sample_rate):
        """
        Makes a MelSpectrogram module
        """
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=4096,
            win_length=4096,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            normalized=True)

def end_to_end(model, hp, source_file, dest_file):
    """
    Runs the contents of `source_file` through the AutoEncoder `model`, writing the 
    reconstructed audio file into `dest_file`. 

    model: AutoEncoder
    hp: Hyperparameters
    source_file: str
    dest_file: str
    """
    print("will load")
    signal, sample_rate = load_and_check(source_file)
    print(f"did load signal: {signal.shape}")
    spec_obj = hp.make_spectrogram_object(sample_rate=sample_rate)
    print("did make spec obj")
    batches = make_batches(signal, hp)
    inference = model(batches, spec_obj)
    inference = inference.view(-1)
    inference = inference.unsqueeze(0)

    torchaudio.save(dest_file, inference, sample_rate)
    print(f"did save to {dest_file}")

def train_model(hp, auto_encoder, training_file_names, validation_file_names):
    """
    The basic idea is that we loop over some unlabeled training examples, and for each file,
    compute the loss and update the autoencoder. Since the autoencoder can only focus on
    a few seconds of audio at a time, we split the entire song (which might be many minutes)
    into batches that cover its duration

    hp: Hyperparameters
    auto_encoder: AutoEncoder
    training_file_names: list[string]
    validation_file_names: list[string]
    """

    model_save_path = ""

    optimizer = torch.optim.AdamW(auto_encoder.parameters(),
                                  lr=hp.learning_rate,
                                  weight_decay=hp.weight_decay)

    # Helper function to compute loss...
    def compute_loss_from_file(file_path):
        signal, sample_rate = load_and_check(file_path)
        spec_obj = hp.make_spectrogram_object(sample_rate=sample_rate)
        batches = make_batches(signal, hp)
        expected = make_expected_from_batches(batches, hp)

        # inference = torch.randn_like(expected)
        inference = auto_encoder(batches, spec_obj)

        # compute the loss between what we expected, and what the model infered.
        # use each spectrogram object in spec_objs to say that the model
        # should also get the spectrograms of expected and inference to look similar
        spec_objs = [spec_obj]
        loss = compute_autoencoder_loss(batches, expected, inference, spec_objs, hp)

        return loss

    should_save_model = False
    for epoch in range(hp.epochs):
        print("Epoch | Batch | Train Loss | Valid Loss")

        random.shuffle(training_file_names)
        total_loss_from_epoch = 0
        training_stats_string = ""
        valid_stats_string = ""

        for batch_index, file_name in enumerate(training_file_names):
            optimizer.zero_grad()
            loss = compute_loss_from_file(file_name)
            loss.backward()
            optimizer.step()
            total_loss_from_epoch += loss

            # print out current training stats, and validate occasionally
            training_stats_string = (
                f"\r{epoch:02d}    "
                f"| {batch_index}/{len(training_file_names)}   "
                f"| {total_loss_from_epoch / (batch_index + 1)}"
            )

            if batch_index > 0 and batch_index % hp.validate_every_n == 0:
                with torch.no_grad():
                    valid_loss = 0
                    for valid_file in validation_file_names:
                        valid_loss += compute_loss_from_file(file_name)
                    valid_stats_string = f"| {valid_loss / len(validation_file_names)}"

            print(training_stats_string + valid_stats_string, end="")

        print("\nTODO: save the model sometimes")

        if should_save_model:
            torch.save(auto_encoder.state_dict(), model_save_path)

    return auto_encoder


def maddy_local_testing():
    training = ["../let_it_go_versions/raw/orchestra.wav"]
    validation = ["../let_it_go_versions/raw/vocals.wav"]
    
    hp = Hyperparameters()
    auto_encoder = AutoEncoder(hp)
    print("finished creating auto_encoder...")

    train_model(hp, auto_encoder, training, validation)
    
    source_file = "../let_it_go_versions/raw/soundtrack.wav"
    dest_file = "../e2e_lig_soundtrack.wav"
    end_to_end(auto_encoder, hp, source_file, dest_file)

maddy_local_testing()
