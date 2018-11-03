import scipy.io.wavfile as wf

def read(filename):
    _, data = wf.read(filename)
    return data
