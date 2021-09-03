import math
import torch
import librosa
from scipy.io.wavfile import read as loadwav
import numpy as np
import datasets

import warnings
warnings.filterwarnings("ignore")

MAX_WAV_VALUE = 32768.0

""" Mel-Spectrogram extraction code from Turab ood_audio"""

# def mel_spectrogram(audio, n_fft, n_mels, hop_length, sample_rate):
#     # Compute mel-scaled spectrogram
#     mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
#     spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
#     mel = np.dot(mel_fb, np.abs(spec))
#
#     # return librosa.power_to_db(mel, ref=0., top_db=None)
#     return np.log(mel + 1e-9)

""" Mel-Spectrogram extraction code from HiFi-GAN meldataset.py"""

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram_hifi(audio, n_fft, n_mels, sample_rate, hop_length, fmin, fmax, center=False):
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    if torch.min(audio) < -1.:
        print('min value is ', torch.min(audio))
    if torch.max(audio) > 1.:
        print('max value is ', torch.max(audio))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_fb = librosa.filters.mel(sample_rate, n_fft, n_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(audio.device)] = torch.from_numpy(mel_fb).float().to(audio.device)
        hann_window[str(audio.device)] = torch.hann_window(n_fft).to(audio.device)

    audio = torch.nn.functional.pad(audio.unsqueeze(1), (int((n_fft-hop_length)/2), int((n_fft-hop_length)/2)), mode='reflect')
    audio = audio.squeeze(1)

    spec = torch.stft(audio, n_fft, hop_length=hop_length, window=hann_window[str(audio.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel = torch.matmul(mel_basis[str(fmax)+'_'+str(audio.device)], spec)
    mel = spectral_normalize_torch(mel).numpy()

    # pad_size = math.ceil(mel.shape[2] / 8) * 8 - mel.shape[2]
    #
    # mel = np.pad(mel, ((0, 0), (0, 0), (0, pad_size)))

    return mel

""" Mel-Spectrogram extraction code from HiFi-GAN meldataset.py"""

class Audio2Mel(torch.utils.data.Dataset):
    def __init__(self, audio_files, max_length, n_fft, n_mels, hop_length, sample_rate, fmin, fmax):
        self.audio_files = audio_files
        self.max_length = max_length # max length of audio
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax

    def __getitem__(self, index):
        filename = self.audio_files[index]
        class_id = datasets.get_class_id(filename)
        salience = datasets.get_salience(filename)

        sample_rate, audio = loadwav(filename)

        audio = audio / MAX_WAV_VALUE
        # audio = (audio - 0.5) * 0.95

        # print('audio length {}'.format(audio.size(1)))

        # error handling
        if sample_rate != self.sample_rate:
            raise ValueError("{} sr doesn't match {} sr ".format(sample_rate, self.sample_rate))

        if len(audio) > self.max_length:
            #raise ValueError("{} length overflow".format(filename))
            audio = audio[0:self.max_length]

        # pad audio to max length, 4s for Urbansound8k dataset
        if len(audio) < self.max_length:
            # audio = torch.nn.functional.pad(audio, (0, self.max_length - audio.size(1)), 'constant')
            audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')

        # mel = mel_spectrogram(audio, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length, sample_rate=self.sample_rate)

        mel_spec = mel_spectrogram_hifi(audio, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length, sample_rate=self.sample_rate, fmin=self.fmin, fmax=self.fmax)



        # print(mel_spec.shape)
        return mel_spec, class_id, salience, filename

    def __len__(self):
        return len(self.audio_files)


    
def extract_flat_mel_from_Audio2Mel(Audio2Mel):
    mel = []

    for item in Audio2Mel:
        mel.append(item[0].flatten())

    return np.array(mel)



if __name__ == '__main__':
    train_file_list, test_file_list = datasets.get_dataset_filelist()

    print(train_file_list[100])

    train_set = Audio2Mel(train_file_list[0:2], 22050 * 4, 1024, 80, 256, 22050, 0, 8000)

    print(train_set[0][0].shape)
