from .toMidi_data import *
from .toMidi_eval import *
from .tokenizer import *
import subprocess
import os
from config.config import *
import torch
import torchaudio
import math
import random
import timeit
from torch import nn
from torch import Tensor


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
                 timeGranularity: float=0.05,
                 maxPredictedTokens: int=100,
                 
                 sample_rate: int=16000,
                 emb_size=512,
                 nhead=8,
                 ffn_hid_dim=512,
                 batch_size=128,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 lr=0.0001,
                 betas=(0.9, 0.98),
                 eps=1e-9,
                 num_epochs=10,
                 dropout: float=0.1,

                 n_fft: int=4000,
                 win_length: int=4000,
                 hop_length: int=2000,
                 n_mels: int = 128):
        
        self.clipLength = clipLength
        self.sample_length = sample_rate * clipLength
        self.timeGranularity = timeGranularity
        self.maxPredictedTokens = maxPredictedTokens
        
        self.sample_rate = sample_rate
        self.frame_rate = self.sample_rate
        self.emb_size = emb_size
        self.nhead = nhead
        self.ffn_hid_dim = ffn_hid_dim
        self.batch_size = batch_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.num_epochs = num_epochs
        self.dropout = dropout
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using CUDA acceleration!")


        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.nHops = 1 + (self.sample_length // self.hop_length)

            
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0)])

class WavEmbedding(nn.Module):
    def __init__(self, sample_length: int, frame_rate: int, emb_size: int,
                 n_fft: int=4000, win_length: int=4000, hop_length: int=2000, n_mels: int = 128):
        super(WavEmbedding, self).__init__()
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=frame_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.ffn = nn.Linear(n_mels, emb_size)

        self.sample_length = sample_length
        self.frame_rate = frame_rate
        self.emb_size = emb_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def forward(self, samples: Tensor):
        spec = self.spectrogram(samples)
        spec = torch.permute(spec, (0, 2, 1))
        out = self.ffn(spec)
        return out

class MidiTokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(MidiTokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        tokens = torch.permute(tokens, (1, 0))
        x = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return x

class Wav2MidiTokenTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 sample_length: int,
                 frame_rate: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float,
                 n_fft: int,
                 win_length: int,
                 hop_length: int,
                 n_mels: int):
        super(Wav2MidiTokenTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_emb = WavEmbedding(sample_length, frame_rate, emb_size,
                                    n_fft, win_length, hop_length, n_mels)
        self.tgt_emb = MidiTokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_emb(tgt))
        src_emb = torch.permute(src_emb, (1, 0, 2))
        tgt_emb = torch.permute(tgt_emb, (1, 0, 2))
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        y = self.generator(outs)
        return y

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_emb(tgt)), memory, tgt_mask)

def create_mask(src, tgt, device, tokenizer, hp):
    src_seq_len = hp.nHops
    tgt_seq_len = tgt.shape[0]


    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = torch.zeros((src.shape[0], src_seq_len))
    tgt_padding_mask = (tgt == tokenizer.padIndex()).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    
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

class Main:
    def __init__(self):
        print("CHECK BATCH ORDER!!!")
        self.hp = Hyperparameters()
        self.tokenizer = Tokenizer(self.hp.clipLength, self.hp.timeGranularity)
        self.transformer = Wav2MidiTokenTransformer(self.hp.num_encoder_layers,
                                                    self.hp.num_decoder_layers,
                                                    self.hp.emb_size,
                                                    self.hp.nhead,
                                                    self.hp.sample_length,
                                                    self.hp.frame_rate,
                                                    self.tokenizer.vocab_size(),
                                                    self.hp.ffn_hid_dim,
                                                    self.hp.dropout,
                                                    self.hp.n_fft,
                                                    self.hp.win_length,
                                                    self.hp.hop_length,
                                                    self.hp.n_mels)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.transformer = self.transformer.to(self.hp.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.padIndex())
        self.optimizer = torch.optim.Adam(self.transformer.parameters(), self.hp.lr, self.hp.betas, self.hp.eps)

    def batchify(self, file_path: str):
        (wav, pitch_intervals) = get_wav_and_pitch_intervals_for_file(file_path, sample_rate=self.hp.sample_rate)

        fractional_n_batches = float(len(wav)) / self.hp.sample_length
        n_batches = int(math.ceil(fractional_n_batches))
        pad_amount = (n_batches * self.hp.sample_length) - len(wav)
        
        src = torch.nn.functional.pad(wav, (0, pad_amount), mode='constant', value=0)
        src = torch.reshape(src, (self.hp.sample_length, -1))
        src = torch.permute(src, (1, 0))
        tgt = self.tokenizer.tokenize(pitch_intervals, batchSize=src.shape[0], toInts=True, padToLength=self.hp.maxPredictedTokens)
        tgt = Tensor(tgt)
        
        tgt = torch.permute(tgt, (1, 0))
        

        return src, tgt
    
    def train_epoch(self):
        self.transformer.train()
        losses = 0
        nBatches = 0

        for training_file in get_training_files():
            print(training_file)
            src, tgt = self.batchify(training_file)

            nBatches += src.shape[0]

            src = src.to(self.hp.device)
            tgt = tgt.to(self.hp.device)

            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.hp.device, self.tokenizer, self.hp)

            print("will pass to transformer")
            logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            print("did pass to transformer")
            self.optimizer.zero_grad()


            tgt_out = tgt[1:, :].long()
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()

        return losses / nBatches

    def evaluate(self):
        self.model.eval()
        losses = 0
        nBatches = 0

        for validation_file in get_validation_files():
            src, tgt = self.batchify(validation_file)
            nBatches += src.shape[0]

            src = src.to(self.hp.device)
            tgt = tgt.to(self.hp.device)

            tgt_input = tgt[:-1, :1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.hp.device, self.tokenizer)

            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / nBatches

    def train(self):
        for epoch in range(1, self.hp.num_epochs+1):
            start_time = timeit.default_timer()
            train_loss = self.train_epoch()
            end_time = timeit.default_timer()
            val_loss = self.evaluate()
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time: {(end_time - start_time):.3f}s")

    def greedy_decode(self, src, src_mask):
        src = src.to(self.hp.device)
        src_mask = src_mask.to(self.hp.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(tokenizer.sosIndex()).type(torch.long).to(self.hp.device)
        for i in range(self.hp.maxPredictedTokens - 1):
            memory = memory.to(self.hp.device)
            tgt_mask = (nn.Transformer.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.hp.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_token = torch.max(prob, dim=1)
            next_token = next_token.item()

            ys = toch.cat([ys,
                           torch.ones(1, 1).type_as(src.data).fill_(next_token)], dim=0)
            if next_token == self.tokenizer.eosIndex():
                break
        return ys

    def transcribe(wavData: Tensor):
        self.model.eval()
        src = wavData
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask).flatten()

        return self.detokenizer(tgt_tokens)
             

def train_toMidi():
    main = Main()
    main.train()

