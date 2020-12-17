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

def lag(signal, prah):
    return np.argmax(np.abs(signal[prah:])) + prah


def autocorr(signal):
    N = signal.size
    tmp = np.zeros(N)
    for k in range(0, N):
        sum = 0
        for n in range(k, N):
            sum += signal[n] * signal[n-k]
        tmp[k] = sum

    return tmp

def dft(x):
    N = len(x)
    X = [sum((x[k] * np.exp(np.complex(0, -2 * math.pi * n * k / N))
            for k in range(N)))
                for n in range(N)]
    return X

data_tone_off, fs = soundfile.read('../audio/maskoff_tone.wav')
startOFF = 27200
tone_off = data_tone_off[startOFF:startOFF+fs]
t1 = np.arange(tone_off.size) / fs

data_tone_on, fs = soundfile.read('../audio/maskon_tone.wav')
startON = 28160
tone_on = data_tone_on[startON:startON+fs]

# startOFF = 0
# startON = 0
# max = 0
# size = 16000
# for j in range(0, data_tone_off.size, 1600):
#     for i in range(0, data_tone_on.size, 320):
#         ccr = np.correlate(data_tone_off[j:j+size], data_tone_on[i:i+size])
#         cmax = np.max(ccr)
#
#         if (cmax > max):
#             max = cmax
#             startOFF = j
#             startON = i
# print(startOFF, startON)
# quit()

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(np.arange(data_tone_off.size), data_tone_off)
# ax2.plot(np.arange(data_tone_on.size), data_tone_on)
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(np.arange(data_tone_off[startOFF:startOFF+16000].size), data_tone_off[startOFF:startOFF+16000])
# ax2.plot(np.arange(data_tone_on[startON:startON+16000].size), data_tone_on[startON:startON+16000])
# plt.show()
# quit()


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

N = 99
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


prah = 20
samples = 1024
base_freqsOFF = np.zeros(N)
base_freqsON = np.zeros(N)
sgrOFF = np.zeros(shape=[N, samples], dtype=complex)
sgrON = np.zeros(shape=[N, samples], dtype=complex)
H = np.zeros(samples, dtype=complex)

# resultB = autocorr(central_clip(ramec_on[5]))
# L = np.argmax(np.abs(resultB[prah:])) + prah
# base_freqsON[5] = fs / L
# plt.figure(999, figsize=(6,3))
# plt.plot(np.arange(resultB.size), resultB)
# plt.show()
# quit()

for i in range(0, N):
    # mask off
    cc = central_clip(ramec_off[i])
    ac = autocorr(cc)
    L = lag(ac, prah)
    base_freqsOFF[i] = fs / L
    # spec
    sgrOFF[i] = np.fft.fft(ramec_off[i], samples)

    # mask on
    resultB = autocorr(central_clip(ramec_on[i]))
    L = np.argmax(np.abs(resultB[prah:])) + prah
    base_freqsON[i] = fs / L
    # spec
    sgrON[i] = np.fft.fft(ramec_on[i], samples)

    # frequency characteristics
    H += sgrON[i] / sgrOFF[i]

sgrOFF_log = np.transpose(10 * np.log10(np.abs(sgrOFF)**2))
sgrON_log = np.transpose(10 * np.log10(np.abs(sgrON)**2))

H = np.abs(H) / N
io = np.fft.ifft(H)

outT = ss.lfilter(H, 1.0, data_tone_off)
outT /= np.abs(outT).max()
wav.write('tone.wav', fs, outT.astype(data_tone_off.dtype))


sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
outS = ss.lfilter(H, 1.0, sentenceOFF)
outS /= np.abs(outS).max()
wav.write('sentence.wav', fs, outS.astype(data_tone_off.dtype))


# PROTOCOL DATA
dir = '../img/'
# --------------------------    DATA
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(t1, tone_off)
# ax2.plot(t2, tone_on)
# --------------------------    [3] RAMCE
x = np.arange(0., ramec_size_s, ramec_size_s / ramec_size)
plt.figure(figsize=(10,4))
plt.plot(x, ramec_off[20], label='mask off')
plt.plot(x, ramec_on[20], label='mask on')

plt.xlabel('vzorky')
plt.suptitle('Rámce')
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(dir + 'ramce.png')

# --------------------------    [4] RAMEC
fig, axs = plt.subplots(4)
fig.set_size_inches(10,7)

axs[0].plot(x, ramec_off[20], label='mask off')
axs[0].set_title('Rámec')
axs[0].set_xlabel('čas')
# --------------------------    [4] CENTRALNE KLIPOVANIE
x = list(range(ramec_size))
cc = central_clip(ramec_off[20])
axs[1].plot(x, cc)
axs[1].set_title('Centrálne klipovanie so 70 %')
axs[1].set_xlabel('vzorky')
# --------------------------    [4] AUTOKORELACIA
ac = autocorr(cc)
lg = lag(ac, prah)
val = ac[lg]
axs[2].plot(x, ac)
axs[2].set_title('Autokorelácia')
axs[2].set_xlabel('vzorky')
axs[2].plot([prah, prah], [0, 20], 'black', lw=2, label='Prah')
axs[2].plot([lg, lg], [0, val], 'red', lw=2, label='Lag')
axs[2].legend()
# --------------------------    [4] ZAKLADNE FREQs
xa = list(range(len(base_freqsOFF)))
xb = list(range(len(base_freqsON)))
axs[3].plot(xa, base_freqsOFF, label='mask off')
axs[3].plot(xb, base_freqsON, label='mask on')
axs[3].set_title('Základné frekvencie rámcov')
axs[3].set_xlabel('rámce')
axs[3].legend()

plt.tight_layout()
plt.savefig(dir + 'baseFqs.png')

# mean & variance of base freqs     [4b]
# print(np.mean(base_freqsOFF), np.var(base_freqsOFF))
# print(np.mean(base_freqsON), np.var(base_freqsON))


# --------------------------    [5] SPECTROGRAMY
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(10, 8)
im1 = ax1.imshow(sgrOFF_log[:int(samples/2)], origin='lower', extent=[0.,1., 0,int(fs/2)], aspect='auto')
im2 = ax2.imshow(sgrON_log[:int(samples/2)], origin='lower', extent=[0.,1., 0,int(fs/2)], aspect='auto')

ax1.set_title('Spectrogram bez rúšky')
ax2.set_title('Spectrogram s rúškou')
ax1.set_xlabel('čas')
ax2.set_xlabel('čas')
ax1.set_ylabel('frekvencia')
ax2.set_ylabel('frekvencia')

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig(dir + 'sgrs.png')


# --------------------------    [6] FREQ CHARAKTERISTIKA FILTRA
plt.figure(figsize=(8,6))
plt.suptitle("Frekvenčná charakteristika rúšky")

plt.plot(np.arange(samples) / samples * fs/2, H)
plt.xlabel('frekvencia')

plt.tight_layout()
plt.savefig(dir + 'freqch.png')
# --------------------------    [7] IMPULZNA ODOZVA
plt.figure(figsize=(8,6))
plt.suptitle("Impulzná odozva rúšky")
plt.plot(np.arange(samples), np.abs(io))
plt.xlabel('čas')

plt.tight_layout()
plt.savefig(dir + 'impOdoz.png')
# --------------------------    [8] VYSLEDOK
fig, axs = plt.subplots(4)
fig.set_size_inches(10, 10)

sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
sentenceON, fs = soundfile.read('../audio/maskon_sentence.wav')
sim = ss.lfilter(H, 1.0, sentenceOFF)
sim /= np.abs(sim).max()

sentenceOFF /= np.abs(sentenceOFF).max()
sentenceON /= np.abs(sentenceON).max()


axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF, label='off')
axs[0].plot(np.arange(sentenceON.size), sentenceON, label='on')
axs[0].legend()

# axs[1].set_title('Veta s rúškou')
# axs[1].plot(np.arange(sentenceON.size), sentenceON)

axs[2].set_title('Veta so simulovanou rúškou')
axs[2].plot(np.arange(sentenceOFF.size), sentenceOFF, label='off')
axs[2].plot(np.arange(sim.size), sim, label='sim')

axs[3].plot(np.arange(sim.size), sim, label='sim')
axs[3].plot(np.arange(sentenceOFF.size), sentenceOFF, label='off')

axs[2].legend()
axs[3].legend()

plt.tight_layout()
plt.savefig(dir + 'outcome.png')
# --------------------------

# plt.show()
