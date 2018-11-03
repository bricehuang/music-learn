import numpy as np
import math

STFT_WINDOW_LEN = 1024/22050.0

def divide(clip, window, hop):
    L = []
    for i in range(0,len(clip),hop):
        L.append(clip[i:i+window])
    return L

def toMono(snippet):
    channels = len(snippet[0])
    if channels == 1:
        # already mono
        return snippet
    elif channels == 2:
        def _average_vals(vals):
            return float(sum(vals)) / len(vals)
        return np.array([_average_channels(vals) for vals in snippet])
    else:
        assert False

def downsample(snippet, rate):
    # snippet is mono
    xs = np.arange(0, len(snippet))
    sample_xs = np.arange(0, len(snippet), 1./rate)
    return np.interp(sample_xs, xs, snippet)

def normalize(snippet):
    # snippet is mono
    rms_amp = (np.dot(snippet, snippet) / len(snippet)) ** 0.5
    return snippet / rms_amp

def melSpectrogram(spectrogram):
    mspect = np.zeros((len(spectrogram), 128))
    for row in range(len(spectrogram)):
        pcol = -1
        for m in range(128):
            f = 700*(10**(m/104) - 1)
            col = f*1.0/STFT_WINDOW_LEN
            assert(col < len(spectrogram[row]))
            mspect[row][m] = sum(spectrogram[row][(pcol+1):(col+1)])/float(col-pcol)
            pcol = col
    return mspect