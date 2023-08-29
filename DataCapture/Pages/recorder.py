import pyaudio
import wave
import os

class Recorder:
    def __init__(self, record_info):
        self.direction, self.gender = record_info
        self.chunks = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_freq = 44100
        self.seconds = 3

    def record(self):
        os.system('find . -name \".DS_Store\" -print -delete')
        self.getFileName()
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

    def getFileName(self):
        path = os.path.join('/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DataCapture/Recordings', self.direction, self.gender)
        files = os.listdir(path)
        self.filename = os.path.join(path, f'{self.direction.lower()}_{self.gender.lower()}_{len(files) + 1}.wav')

    def save(self):
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.sample_freq)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print("Audio saved.")