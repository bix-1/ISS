import soundfile
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as ss
import scipy.io.wavfile as wav

def central_clip(signal):
    data = np.array(signal).copy()
    split = max(np.abs(signal)) * 0.7
    top = signal > split
    bottom = signal < -split
    rest = np.where(np.logical_and(signal >= -split, signal <= split))

    data[top] = 1
    data[bottom] = -1
    data[rest] = 0

    return data

def autocorr(signal):
    N = signal.size
    tmp = np.zeros(N)
    for k in range(0, N):
        sum = 0
        for n in range(0, N-k):
            sum += signal[n] * signal[n+k]
        tmp[k] = sum

    return tmp

def dft(x):
    N = len(x)
    X = [sum((x[k] * np.exp(np.complex(0, -2 * math.pi * n * k / N))
            for k in range(N)))
                for n in range(N)]
    return X

data_tone_off, fs = soundfile.read('../audio/maskoff_tone.wav')
tone_off = data_tone_off[35000:51000]
t1 = np.arange(tone_off.size) / fs

data_tone_on, fs = soundfile.read('../audio/maskon_tone.wav')
tone_on = data_tone_on[32000:48000]
t2 = np.arange(tone_on.size) / fs

t1 -= np.mean(t1)
t2 -= np.mean(t2)

t1 /= np.abs(t1).max()
t2 /= np.abs(t2).max()

# ramce
ramec_size_s = 0.02
freq = 16000
ramec_size = int(ramec_size_s * freq)
step = int(0.01 * freq)

N = 100
ramec_off = []
ramec_on = []

for i in range(0, N):
    start = i * step
    tmp1 = tone_off[start : start + ramec_size]
    tmp2 = tone_on[start : start + ramec_size]
    ramec_off.append([])
    ramec_on.append([])
    ramec_off[i] = tmp1.copy()
    ramec_on[i] = tmp2.copy()
# TODO handle posledny ramec

# ramecA = central_clip(ramec_off[20])
# resultA = autocorr(ramecA)
# ramecB = central_clip(ramec_on[20])
# resultB = autocorr(ramecB)

prah = 10
samples = 1024
base_freqsA = np.zeros(N)
base_freqsB = np.zeros(N)
sgrOFF = np.zeros(shape=[N, samples], dtype=complex)
sgrON = np.zeros(shape=[N, samples], dtype=complex)
H = np.zeros(samples, dtype=complex)
for i in range(0, N):
    # mask off
    resultA = autocorr(central_clip(ramec_off[i]))
    indexA = np.argmax(np.abs(resultA[prah:])) + prah
    tmp = np.abs(ramec_off[i][indexA])
    base_freqsA[i] = tmp * 1600
    # spec
    sgrOFF[i] = np.fft.fft(ramec_off[i], samples)

    # mask on
    resultB = autocorr(central_clip(ramec_on[i]))
    indexB = np.argmax(np.abs(resultB[prah:])) + prah
    tmp = np.abs(ramec_off[i][indexB])
    base_freqsB[i] = tmp * 1600
    # spec
    sgrON[i] = np.fft.fft(ramec_on[i], samples)

    # frequency characteristics
    H += sgrON[i] / sgrOFF[i]

sgrOFF_log = np.transpose(10 * np.log10(np.abs(sgrOFF)**2))
sgrON_log = np.transpose(10 * np.log10(np.abs(sgrON)**2))

H = np.abs(H) / N
io = np.fft.ifft(H)

out = ss.lfilter(H, 1.0, data_tone_off)
out /= np.abs(out).max()
wav.write('tone.wav', fs, out.astype(data_tone_off.dtype))


sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
out = ss.lfilter(H, 1.0, sentenceOFF)
out /= np.abs(out).max()
wav.write('sentence.wav', fs, out.astype(data_tone_off.dtype))

# print(np.mean(base_freqsA), np.var(base_freqsA))
# print(np.mean(base_freqsB), np.var(base_freqsB))


# --------------------------    DATA
# plt.figure(1, figsize=(6,3))
# plt.plot(t1, tone_off)
# plt.figure(2, figsize=(6,3))
# plt.plot(t2, tone_on)
# --------------------------    RAMCE
# TODO add to protocol
# plt.figure(2, figsize=(6,3))
# x = list(range(ramec_size))
# plt.plot(x, ramec_off[20])
# plt.plot(x, ramec_on[20])
# --------------------------    CENTRALNE KLIPOVANIE
# TODO grafy z 4.a
# x = list(range(ramec_size))
# plt.figure(3, figsize=(6,3))
# plt.plot(x, ramec_off[20])
# plt.plot(x, ramecA)
# plt.figure(4, figsize=(6,3))
# plt.plot(x, ramec_on[20])
# plt.plot(x, ramecB)
# --------------------------    AUTOKORELACIA
# plt.figure(5, figsize=(6,3))
# plt.plot(t1[:ramec_size], resultA)
# plt.plot(t2[:ramec_size], resultB)
# --------------------------    ZAKLADNE FREQs
# plt.figure(6, figsize=(6,3))
# xa = list(range(len(base_freqsA)))
# xb = list(range(len(base_freqsB)))
# plt.plot(xa, base_freqsA)
# plt.plot(xb, base_freqsB)
# --------------------------    SPECTROGRAMS
# plt.figure(8, figsize=(9,3))
# plt.imshow(sgrOFF_log[:int(samples/2)], origin='lower', extent=[0.,1., 0,int(fs/2)], aspect='auto')
# plt.colorbar()
# plt.figure(9, figsize=(9,3))
# plt.imshow(sgrON_log[:int(samples/2)], origin='lower', extent=[0.,1., 0,int(fs/2)], aspect='auto')
# plt.colorbar()
# --------------------------    FREQ CHARAKTERISTIKA FILTRA
# plt.figure(10, figsize=(6,3))
# plt.plot(np.arange(samples), H)
# --------------------------    IMPULZNA ODOZVA
# plt.figure(11, figsize=(6,3))
# plt.plot(np.arange(samples), io)
# --------------------------
# plt.figure(12, figsize=(6,3))
# plt.plot(np.arange(data_tone_off.size), out)
# plt.plot(np.arange(data_tone_on.size), data_tone_on)
# --------------------------



plt.show()
