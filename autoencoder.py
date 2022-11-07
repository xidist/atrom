import torch
import torchaudio
import math

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


def make_batches_with_source_and_context(signal, left_context_width, source_width, right_context_width, source_start_times):
    """
    signal: Tensor[frame_length] containing the audio data
    left_context_width: int. the number of samples before source_start_times[i] that
                        should be included in a batch
    source_width: int. the number of samples, starting at source_start_times[i], that
                  should be included in a batch
    right_context_width: int. the number of samples after the end of source_width that
                         should be included in a batch
    source_start_times: Tensor[int] containing the times at which each batch should start

    returns: Tensor[batch_size x (left_context_width + source_width + right_context_width)].
             A tensor of batches, where each batch is a contiguous slice of audio data. 
             May be left or right padded with zeros at its ends
    """
    signal = torch.nn.functional.pad(signal, (left_context_width, source_width + right_context_width),
                                     "constant", 0)
    subscript_length = left_context_width + source_width + right_context_width

    # make a [batch x subscript_length] tensor for indexing signal
    indices = torch.arange(start=0, end=subscript_length, step=1, dtype=int)
    indices = indices.unsqueeze(0).repeat(source_start_times.shape[0], 1)
    # make a [batch x subscript_length] tensor for shifting indices
    offset = source_start_times.unsqueeze(1).repeat(1, subscript_length)
    # add source_start_times[i] to each indices[i]
    indices += offset

    return signal[indices]

def make_expected_from_batches(batches, left_context, source, right_context):
    """
    batches: Tensor[batch_count x left_context + source + right_context]. Audio signals
             for the autoencoder to reproduce
    left_context: int. The number of left context samples
    source: int. The number of source samples
    right_context: int. The number of source samples

    returns: Tensor[batch_count x source]. The center slice of each batch we want
             the autoencoder to reproduce
    """
    # make a [source] tensor for subscripting batches
    indices = torch.arange(start=0, end=source, step=1, dtype=int) + left_context

    result =  batches[:, indices]
    return result

def put_inference_inside_batches(batches, inference, left_context, source, right_context):
    """
    batches: Tensor[batch_count x left_context + source + right_context]. Audio signals
             for the autoencoder to reproduce
    inference: Tensor[batch_count x source]. The audio signal the autoencoder actually reproduced
    left_context: int. The number of left context samples
    source: int. The number of source samples
    right_context: int. The number of source samples

    returns: Tensor[batch_count x left_context + source + right_context]. A copy of batches,
             with the source samples replaced with the contents of inference
    """
    b_copy = batches.clone()
    # make a [source] tensor for subscripting b_copy
    indices = torch.arange(start=0, end=source, step=1, dtype=int) + left_context
    b_copy[:, indices] = inference
    return b_copy

def make_spectrogram_object(sample_rate):
    """
    Factory method to return a MelSpectrogram module
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=4096,
        win_length=4096,
        hop_length=2048,
        n_mels=128,
        window_fn=torch.hann_window,
        normalized=True)

def compute_autoencoder_loss(batches, expected, inference, spec_objs, left_context, source, right_context):
    """
    batches: Tensor[batch_count x left_context + source + right_context]. Raw audio signal
    expected: Tensor[batch_count x source]
    inference: Tensor[batch_count x source]
    spec_objs: A list of objects capable of computing a spectrogram. 
               The MSE will be computed using each spectrogram object, comparing the 
               spectrogram between using batches, and the spectrogram putting inference
               in the middle of batches
    left_context: int. The number of left context samples
    source: int. The number of source samples
    right_context: int. The number of source samples

    returns: float. The loss of the autoencoder
    """

    # note: if inference is drawn from a normal distribution,
    # loss from wave-to-wave comparison is 1,
    # and loss from spectrogram comparison is 158.
    # do we need to find a way to balance these to the same order of magnitude?
    
    loss = torch.nn.functional.mse_loss(inference, expected)
    batches_with_inference = put_inference_inside_batches(batches, inference, left_context,
                                                          source, right_context)
    
    for spec_obj in spec_objs:
        inf_spec = spec_obj(batches_with_inference)
        exp_spec = spec_obj(batches)
        loss += torch.nn.functional.mse_loss(inf_spec, exp_spec);

    return loss

class AutoEncoder(torch.nn.Module):
    def __init__(self, left_context, source, right_context, spec_obj):
        """
        left_context: int. The number of left context samples
        source: int. The number of source samples
        right_context: int. The number of source samples
        spec_obj: The object that should be used to compute the spectrogram
        """

        self.left_context = left_context
        self.source = source
        self.right_context = right_context
        self.spec_obj = spec_obj

        # todo: replace these magic numbers with good code
        n_mels = 128
        spec_length = 64

        nonlinear = torch.nn.LeakyReLU()

        self.encoder = torch.nn.Sequential(
            spec_obj,
            torch.nn.Linear(n_mels * spec_length, 4096),
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
            torch.nn.Linear(32768, source)
        )

    def forward(self, x):
        """
        x: Tensor[batch_count x left_context + source + right_context]

        returns: Tensor[batch_count x source]
        """

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

def main():
    """
    The basic idea is that we loop over some unlabeled training examples, and for each file,
    compute the loss and update the autoencoder. Since the autoencoder can only focus on
    a few seconds of audio at a time, we split the entire song (which might be many minutes)
    into batches that cover its duration
    """

    # Hyperparameters...
    left_context = 44100
    source = 44100
    right_context = 44100
    batch_offset = 44100
    
    learning_rate = 1e-3
    weight_decay = 1e-5
    epochs = 10
    validate_every_n = 10

    training_file_names = []
    validation_file_names = []
    model_save_path = ""

    # Create the model...
    auto_encoder = AutoEncoder(left_context, source, right_context)
    optimizer = torch.optim.AdamW(auto_encoder.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)

    # Helper function to compute loss...
    def compute_loss_from_file(file_path):
        signal, sample_rate = load_and_check(file)
        # Computing a spectrogram has a lot of fiddly parameters for setup,
        # so that's hidden away in a helper function
        spec_obj = make_spectrogram_object(sample_rate=sample_rate)

        # each batch should start at batch_offset samples further into the signal
        start_times = torch.arange(start=0, end=signal.shape[0], step=batch_offset, dtype=int)
        # get the batched input we give to the autoencoder
        batches = make_batches_with_source_and_context(signal, left_context, source, right_context, start_times)
        # get the expected output we want it to produce
        expected = make_expected_from_batches(batches, left_context, source, right_context)

        # inference = torch.randn_like(expected)
        inference = auto_encoder(batches)

        # compute the loss between what we expected, and what the model infered.
        # use each spectrogram object in spec_objs to say that the model
        # should also get the spectrograms of expected and inference to look similar
        spec_objs = [spec_obj]
        loss = compute_autoencoder_loss(batches, expected, inference, spec_objs,
                                        left_context, source, right_context)
        return loss

    should_save_model = False
    for epoch in epochs:
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
                "\r {epoch:02d}"
                "| {batch_index}/{len(training_file_names)}"
                "| {total_loss_from_epoch / (batch_index + 1)}"
            )

            if batch_index > 0 and batch_index % validate_every_n == 0:
                with torch.no_grad():
                    valid_loss = 0
                    for valid_file in validation_file_names:
                        valid_loss += compute_loss_from_file(file_name)
                    valid_stats_string = "| {valid_loss / len(validation_file_names)}"

            print(training_stats_string + valid_stats_string, end="")

        print("TODO: save the model sometimes")

        if should_save_model:
            torch.save(auto_encoder.state_dict(), model_save_path)
