import torch
import torchaudio

def chop_waveform_into_examples(waveform, hp):
    """
    waveform: Tensor[frame_length] containing the waveform data
    hp: Hyperparameters

    returns: Tensor[batch_size x hp.example_size]
             A tensor of contiguous slices of the input audio waveform
    """

    # contains the first index of each example
    example_start_times = torch.arange(start=0, end=waveform.shape[0],
                                       step=hp.batch_offset, dtype=int)

    # pad the end so we don't subscript out of bounds
    waveform = torch.nn.functional.pad(waveform, (0, hp.example_size),
                                       "constant", 0)

    # the base unit for subscripting on example
    indices = torch.arange(start=0, end=hp.example_size, step=1, dtype=int)

    # loop the base unit in the batch dimension
    indices = indices.unsqueeze(0).repeat(example_start_times.shape[0], 1)

    # loop the start times in the time dimension
    example_start_times = example_start_times.unsqueeze(1).repeat(1, hp.example_size)

    # each example is now a contiguous slice starting at a different place
    indices += example_start_times

    return waveform[indices]

def make_batches(waveform, hp):
    """
    waveform: Tensor[frame_length] containing the waveform data
    hp: Hyperparameters

    returns: Tensor[batch_size x spectrogram_shape...]
             A tensor of batches, where each batch is a spectrogram of a contiguous
             slice of audio data. 
    """
    return hp.forward_spectrogram(chop_waveform_into_examples(waveform, hp))

def reconstruct_waveform(spec, hp):
    """
    spec: Tensor[batch_size x spectrogram_shape...]
          A tensor of batches, where each batch is a spectrogram

    returns: Tensor[batch_size x example_size]
             The reconstructed waveform
    """
    return hp.backward_spectrogram(torch.abs(spec))

class Hyperparameters:
    """
    Bag of hyperparameters that need to be passed around at times
    """
    def __init__(self,
                 sample_rate = 16000,
                 example_size = 16000,
                 batch_offset = 16000,
                 batch_size = 8,
                 learning_rate = 1e-4,
                 weight_decay = 1e-5,
                 epochs = 10,
                 validate_every_n_batches = 3600,
                 demo_every_n_epochs = 1,
                 n_fft = 4000,
                 win_length = 4000,
                 hop_length = 2000,
                 transformer_num_layers = 6,
                 transformer_nhead = 8):

        """
        Transformer:
            d_model: the number of expected features in the encoder/decoder inputs
                n_fft?
            nhead: number of heads in multiheadattention models
                default 8
            num_encoder_layers: sub-encoder-layer count
                default 6
            num_decoder_layers: sub-decoder-layer count
                default 6
            dim_feedforward: feedforward dimension
                default 2048
            dropout: default 0.1
            activation: default relu
            custom_encoder: default None
            custom_decoder: default None
            layer_norm_eps: default 1e-5
            batch_first: True
            norm_first: default False

        TransformerEncoder: 
            encoder_layer: TransformerEncoderLayer
            num_layers: int
            norm: layer normalization component
                ???
            enable_nested_tensor: False

        TransformerEncoderLayer:
            d_model: d_model
            nhead: nhead
            dim_feedforward: default 2048
            dropout: default 0.1
            activation: default relu
            layer_norm_eps: default 1e-5
            batch_first: True
            norm_first: default False
        """

        self.sample_rate = sample_rate
        self.example_size = example_size
        self.batch_offset = batch_offset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = 10
        self.validate_every_n_batches = validate_every_n_batches
        self.demo_every_n_epochs = demo_every_n_epochs
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.transformer_num_layers = 6
        self.transformer_nhead = 8


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using CUDA acceleration!")

        self.forward_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            power = 2
        ).to(self.device)

        self.backward_spectrogram = torchaudio.transforms.GriffinLim(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length
        ).to(self.device)
