import struct
import wave

def parse_wav_file(filename):
    # type: (str) -> {data: List[int], framerate: int}
    """
        Parses a sound file into a waveform (int array) and framerate.
    """
    w = wave.open(filename, mode='rb')
    data = [struct.unpack('<i',w.readframes(1))[0] for i in xrange(w.getnframes())]
    w.close()
    return {
        'data': data,
        'framerate': w.getframerate(),
    }
result = parse_wav_file('./IRMAS-TrainingData/pia/[pia][jaz_blu]1532__1.wav')
data = result['data']
framerate = result['framerate']
print len(data)
