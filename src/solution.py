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

def overlapAdd(x, h, N):
    M = 1024
    size = x[0].size
    step = int(size / 2)
    out = np.zeros(step * N + M, dtype=complex)

    for i in range(0,N):
        pos = step * i
        ramec = np.zeros(M, dtype=complex)
        ramec[:size] = x[i]
        f = ss.lfilter(h, [1], ramec)
        # f = np.fft.ifft(np.fft.fft(ramec, M) * h)
        out[pos:pos+M] = out[pos:pos+M] + f

    return out

def split(x, N, size, step):
    out = []
    for i in range(0, N):
        start = i * step
        tmp = x[start : start + size]
        out.append([])
        out[i] = tmp.copy()

    return out


data_tone_off, fs = soundfile.read('../audio/maskoff_tone.wav')
data_tone_off = data_tone_off[13200:63300]
startOFF = 14400
tone_off = data_tone_off[startOFF:startOFF+fs]
t1 = np.arange(tone_off.size) / fs

data_tone_on, fs = soundfile.read('../audio/maskon_tone.wav')
data_tone_on = data_tone_on[11400:60600]
startON = 5760
tone_on = data_tone_on[startON:startON+fs]
t2 = np.arange(tone_on.size) / fs

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


t1 -= np.mean(t1)
t2 -= np.mean(t2)

t1 /= np.abs(t1).max()
t2 /= np.abs(t2).max()

# ramce
N = 99
ramec_size_s = 0.02
freq = 16000
ramec_size = int(ramec_size_s * fs)
step = int(0.01 * fs)

ramec_off = split(tone_off, N, ramec_size, step)
ramec_on = split(tone_on, N, ramec_size, step)




prah = 20
samples = 1024
base_freqsOFF = np.zeros(N)
base_freqsON = np.zeros(N)
sgrOFF = np.zeros(shape=[N, samples], dtype=complex)
sgrON = np.zeros(shape=[N, samples], dtype=complex)
H = np.zeros(samples, dtype=complex)
for i in range(0, N):
    # mask off
    cc = central_clip(ramec_off[i])
    ac = autocorr(cc)
    L = lag(ac, prah)
    base_freqsOFF[i] = fs / L

    wOFF = ramec_off[i] * np.hamming(ramec_size)

    # spec
    # sgrOFF[i] = np.fft.fft(ramec_off[i], samples)
    sgrOFF[i] = np.fft.fft(wOFF, samples)

    # mask on
    resultB = autocorr(central_clip(ramec_on[i]))
    L = np.argmax(np.abs(resultB[prah:])) + prah
    base_freqsON[i] = fs / L

    wON = ramec_on[i] * np.hamming(ramec_size)

    # spec
    sgrON[i] = np.fft.fft(wON, samples)

    # frequency characteristics
    H += sgrON[i] / sgrOFF[i]


sgrOFF_log = np.transpose(10 * np.log10(np.abs(sgrOFF)**2))
sgrON_log = np.transpose(10 * np.log10(np.abs(sgrON)**2))


H = np.abs(H) / N
io = np.fft.ifft(H)

outT = ss.lfilter(H, [1], data_tone_off)
outT /= np.abs(outT).max()
wav.write('tone.wav', fs, outT.astype(data_tone_off.dtype))

step = int(ramec_size / 2)
N = int(data_tone_off.size / step) - 1
ramce = split(data_tone_off, N, ramec_size, step)
outTov = overlapAdd(ramce, io, N)
outTov /= np.abs(outTov).max()
wav.write('ov_tone.wav', fs, outTov.astype(data_tone_off.dtype))

sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
step = int(ramec_size / 2)
N = int(sentenceOFF.size / step) - 1
ramce = split(sentenceOFF, N, ramec_size, step)
outS = overlapAdd(ramce, io, N)
outS /= np.abs(outS).max()
wav.write('sentence.wav', fs, outS.astype(data_tone_off.dtype))

# PROTOCOL DATA
dir = '../img/'
# --------------------------    DATA
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(np.arange(data_tone_off.size), data_tone_off)
# ax2.plot(np.arange(data_tone_on.size), data_tone_on)


# --------------------------    TONES
fig, axs = plt.subplots(4)

data_tone_off /= np.abs(data_tone_off).max()
data_tone_on /= np.abs(data_tone_on).max()

axs[0].plot(np.arange(data_tone_off.size), data_tone_off, )
axs[1].plot(np.arange(data_tone_on.size), data_tone_on)
axs[2].plot(np.arange(outT.size), outT)
axs[3].plot(np.arange(outTov.size), outTov)

axs[0].set_title('Mask off')
axs[1].set_title('Mask on')
axs[2].set_title('Sim')
axs[3].set_title('Sim overlap-add')
plt.tight_layout()
plt.savefig(dir + 'tones.png')


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

plt.plot(np.arange(H.size) / samples * fs/2, H)
plt.xlabel('frekvencia')

plt.tight_layout()
plt.savefig(dir + 'freqch.png')
# --------------------------    [7] IMPULZNA ODOZVA
plt.figure(figsize=(8,6))
plt.suptitle("Impulzná odozva rúšky")
plt.plot(np.arange(io.size), io.real)
plt.xlabel('čas')

plt.tight_layout()
plt.savefig(dir + 'impOdoz.png')
# --------------------------    [8] VYSLEDOK
# plt.close('all')
fig, axs = plt.subplots(5)
fig.set_size_inches(10, 10)

sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
sentenceON, fs = soundfile.read('../audio/maskon_sentence.wav')

step = int(ramec_size / 2)
N = int(sentenceOFF.size / step) - 1
ramce = split(sentenceOFF, N, ramec_size, step)
sim = overlapAdd(ramce, io, N)
# sim = ss.lfilter(H, 1.0, sentenceOFF)
sim /= np.abs(sim).max()

sentenceOFF /= np.abs(sentenceOFF).max()
sentenceON /= np.abs(sentenceON).max()


axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF, label='off')
axs[0].legend()

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)

axs[2].set_title('Veta so simulovanou rúškou')
axs[2].plot(np.arange(sim.size), sim)

axs[3].plot(np.arange(sentenceOFF.size), sentenceOFF, label='off')
axs[3].plot(np.arange(sim.size), sim, label='sim')

axs[4].plot(np.arange(sim.size), sim, label='sim')
axs[4].plot(np.arange(sentenceOFF.size), sentenceOFF, label='off')

axs[3].legend()
axs[4].legend()

plt.tight_layout()
plt.savefig(dir + 'outcome.png')
# --------------------------

# plt.show()
