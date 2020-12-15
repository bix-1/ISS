import soundfile
import matplotlib.pyplot as plt
import numpy as np
import math

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
    # tmp = np.correlate(signal, signal, mode='full')
    # return tmp[int(tmp.size/2):]

    N = signal.size
    tmp = np.zeros(N)
    for k in range(0, N):
        sum = 0
        for n in range(0, N-k):
            sum += signal[n] * signal[n+k]
        tmp[k] = sum

    return tmp


data_tone_off, fs_tone_off = soundfile.read('../audio/maskoff_tone.wav')
data_tone_off = data_tone_off[:15999]
t1 = np.arange(data_tone_off.size) / fs_tone_off

data_tone_on, fs_tone_on = soundfile.read('../audio/maskon_tone.wav')
data_tone_on = data_tone_on[:15999]
t2 = np.arange(data_tone_on.size) / fs_tone_on

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
    tmp1 = data_tone_off[start : start + ramec_size]
    tmp2 = data_tone_on[start : start + ramec_size]
    ramec_off.append([])
    ramec_on.append([])
    ramec_off[i] = tmp1
    ramec_on[i] = tmp2
# TODO handle posledny ramec

ramecA = central_clip(ramec_off[20])
resultA = autocorr(ramecA)
ramecB = central_clip(ramec_on[20])
resultB = autocorr(ramecB)

base_freqsA = []
base_freqsB = []
for i in range(0, N):
    prah = 10
    if (ramec_off[i].size > 10): # TODO remove
        resultA = autocorr(central_clip(ramec_off[i]))
        indexA = np.argmax(np.abs(resultA[prah:])) + prah
        tmp = ramec_off[i][indexA]
        base_freqsA.append(np.abs(tmp) * 1600)

    if (ramec_on[i].size > 0): # TODO remove
        resultB = autocorr(central_clip(ramec_on[i]))
        indexB = np.argmax(np.abs(resultB[prah:])) + prah
        tmp = ramec_off[i][indexB]
        base_freqsB.append(np.abs(tmp) * 1600)

# print(np.mean(base_freqsA), np.var(base_freqsA))
# print(np.mean(base_freqsB), np.var(base_freqsB))


# --------------------------    DATA
# plt.figure(1, figsize=(6,3))
# plt.plot(t1, data_tone_off)
# plt.plot(t2, data_tone_on)
# --------------------------    RAMCE
# TODO add to protocol
# plt.figure(2, figsize=(6,3))
# plt.plot(t1[:ramec_size], ramec_off[20])
# plt.plot(t2[:ramec_size], ramec_on[20])
# --------------------------    CENTRALNE KLIPOVANIE
# TODO grafy z 4.a
# plt.figure(3, figsize=(6,3))
# plt.plot(t1[:ramec_size], ramec_off[20])
# plt.plot(t1[:ramec_size], ramecA)
# plt.figure(4, figsize=(6,3))
# plt.plot(t1[:ramec_size], ramec_on[20])
# plt.plot(t1[:ramec_size], ramecB)
# --------------------------    AUTOKORELACIA
# plt.figure(5, figsize=(6,3))
# plt.plot(t1[:ramec_size], resultA)
# plt.plot(t2[:ramec_size], resultB)
# --------------------------
plt.figure(6, figsize=(6,3))
plt.plot(range(0,82), base_freqsA)
plt.plot(range(0,79), base_freqsB)
# --------------------------
plt.show()
