from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pyaudio
import torch
import wave
import os

AUDIO_FILE = '/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Verifier/Inputs/input.wav'
IMAGE_FILE = '/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Verifier/Inputs/spectrogram.png'


class Recorder:
    def __init__(self):
        self.chunks = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_freq = 44100
        self.seconds = 3

    def record(self):
        os.system('find . -name \".DS_Store\" -print -delete')
        self.p = pyaudio.PyAudio()
        stream = self.p.open(format = self.sample_format, channels = self.channels, rate = self.sample_freq, frames_per_buffer = self.chunks, input = True)
        self.frames = []
        for _ in range(0, int(self.sample_freq / self.chunks * self.seconds)):
            data = stream.read(self.chunks)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        self.p.terminate()
        self.save()

    def save(self):
        wf = wave.open(AUDIO_FILE, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.sample_freq)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print("Audio saved.")

class Spectrogramer:
    def __init__(self):
        pass

    def create_spectrogram(self):
        a = read("/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Verifier/Inputs/input.wav")
        a = np.array(a[1],dtype=float)
        plt.figure(figsize = (4.96, 3.69), dpi = 100)
        spectrum, freqs, times, image = plt.specgram(x = a, Fs = 44100, NFFT = 1024)
        plt.ylim(0, 5000)
        cb = plt.colorbar()
        plt.clim(55, -135)
        plt.axis("off")
        cb.remove()

        plt.savefig(fname = IMAGE_FILE, bbox_inches = "tight", pad_inches = 0, dpi = 100)
        image = np.array(Image.open(IMAGE_FILE).convert('RGB'))
        im_resize = torch.tensor(image.reshape((1, image.shape[2], image.shape[0], image.shape[1])))
        im_resize = im_resize.float()
        return im_resize
