import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import read
import numpy as np
import os

root = os.getcwd()

audio_path = os.path.join(root, "Recordings")
spec_path = os.path.join(root, "Spectrograms")

dirs = ["Up", "Down", "Left", "Right"]
subdirs = ["Male", "Female"]

os.system('find . -name \".DS_Store\" -print -delete')

for dir in dirs:
    for gender in subdirs:
        print(f"{dir}/{gender}")
        curr_spec_path = os.path.join(spec_path, dir, gender)
        curr_audio_path = os.path.join(audio_path, dir, gender)
        for audio_file in sorted(os.listdir(curr_audio_path)):
            spectrogram_filename = audio_file[:-4] + ".png"
            if os.path.exists(os.path.join(curr_spec_path, spectrogram_filename)):
                continue

            a = read(os.path.join(curr_audio_path, audio_file))
            a = np.array(a[1],dtype=float)

            plt.figure(figsize = (4.96, 3.69), dpi = 100)
            spectrum, freqs, times, image = plt.specgram(x = a, Fs = 44100, NFFT = 1024)
            plt.ylim(0, 5000)
            cb = plt.colorbar()
            plt.clim(55, -135)
            plt.axis("off")
            cb.remove()
            plt.savefig(fname = os.path.join(curr_spec_path, spectrogram_filename), bbox_inches = "tight", pad_inches = 0, dpi = 100)
            plt.close()

# x_path = os.path.join(audio_path, "x")
# for audio_file in sorted(os.listdir(x_path)):
#     curr_spec_path = os.path.join(spec_path, x_path)
#     curr_audio_path = os.path.join(x_path, audio_file)
#     print(audio_file)
#     a = read(os.path.join(curr_audio_path, audio_file))
#     a = np.array(a[1],dtype=float)

#     spectrum, freqs, times, image = plt.specgram(x = a, Fs = 44100, NFFT = 1024)
#     plt.ylim(0, 5000)
#     plt.show()
