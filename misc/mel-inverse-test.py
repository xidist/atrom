import torch
import torchaudio
import torchvision
import matplotlib.pyplot as plt

def load(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    waveform = torch.sum(waveform, dim=0).squeeze()
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    return waveform

def mel():
    read_path = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio/lig_soundtrack.wav"
    write_path = "/Users/msa/Desktop/out.wav"

    waveform = load(read_path)
    print(f"did load: {waveform.shape}")

    forward = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft = 4000,
        win_length = 4000,
        hop_length = 2000,
        n_mels = 128,
        
    )
    backward_one = torchaudio.transforms.InverseMelScale(
        n_stft = 4000,
        n_mels = 128,
        sample_rate = 16000
    )
    backward_two = torchaudio.transforms.GriffinLim(
        n_fft = 4000,
        win_length = 4000,
        hop_length = 2000
    )

    waveform = forward(waveform)
    print(f"did forward: {waveform.shape}")
    waveform = backward_one(waveform)
    print(f"did backward_one: {waveform.shape}")
    waveform = backward_two(waveform)
    print(f"did backward_two: {waveform.shape}")

    torchaudio.save(write_path, waveform, 16000)
    print(f"did save")

def not_mel():
    read_path = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio/lig_soundtrack.wav"
    write_path = "/Users/msa/Desktop/out.wav"

    waveform = load(read_path)
    print(f"did load: {waveform.shape}")

    forward = torchaudio.transforms.Spectrogram(
        n_fft = 4000,
        win_length = 4000,
        hop_length = 2000,
        power = 2,
    )

    backward = torchaudio.transforms.GriffinLim(
        n_fft = 4000,
        win_length = 4000,
        hop_length = 2000,
    )

    waveform = forward(waveform)
    print(f"did forward: {waveform.shape}")
    waveform = backward(waveform)
    print(f"did backward: {waveform.shape}")

    waveform = waveform.unsqueeze(0)
    torchaudio.save(write_path, waveform, 16000)
    print(f"did save")

def complex():
    read_path = "/Users/msa/Desktop/Penn/Fall 2022/CIS 4000/foobar-maddy/input_audio/lig_soundtrack.wav"
    write_path = "/Users/msa/Desktop/out.wav"

    waveform = load(read_path)
    print(f"did load: {waveform.shape}")

    forward = torchaudio.transforms.Spectrogram(
        n_fft = 4000,
        win_length = 4000,
        hop_length = 2000,
        power = None,
    )

    backward = torchaudio.transforms.InverseSpectrogram(
        n_fft = 4000,
        win_length = 4000,
        hop_length = 2000,
    )

    spec = forward(waveform)
    print(f"did forward: {spec.shape}")

    rand_angle = torch.polar(torch.ones_like(spec, dtype=float), torch.rand(spec.shape, dtype=float)).to(torch.complex64)
    print(spec.dtype)
    print(rand_angle.dtype)

    if True: 
        reconstructed = backward(spec * rand_angle)
        print(f"did backward: {reconstructed.shape}")

        reconstructed = reconstructed.unsqueeze(0)
        torchaudio.save(write_path, reconstructed, 16000)
        print(f"did save")
    

# complex()
not_mel()
