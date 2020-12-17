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


'''__________SEGMENTS__________'''
N = 99
sgmts_size_s = 0.02
sgmts_size = int(sgmts_size_s * fs)
step = int(0.01 * fs)

sgmts_off = split(tone_off, N, sgmts_size, step)
sgmts_on = split(tone_on, N, sgmts_size, step)



prah = 20
samples = 1024
base_freqsOFF = np.zeros(N)
base_freqsON = np.zeros(N)
sgrOFF = np.zeros(shape=[N, samples], dtype=complex)
sgrON = np.zeros(shape=[N, samples], dtype=complex)
H = np.zeros(samples, dtype=complex)
# for variation w/ window func
w_sgrOFF = np.zeros(shape=[N, samples], dtype=complex)
w_sgrON = np.zeros(shape=[N, samples], dtype=complex)
w_H = np.zeros(samples, dtype=complex)

for i in range(0, N):
    #  get base freqs

    # mask off
    cc = central_clip(sgmts_off[i])
    ac = autocorr(cc)
    L = lag(ac, prah)
    base_freqsOFF[i] = fs / L
    # mask on
    cc = central_clip(sgmts_on[i])
    ac = autocorr(cc)
    L = lag(ac, prah)
    base_freqsON[i] = fs / L

    # apply window function
    wOFF = sgmts_off[i]
    wON = sgmts_on[i] * np.hamming(sgmts_size)

    # get spectrograms
    sgrOFF[i] = np.fft.fft(sgmts_off[i], samples)
    sgrON[i] = np.fft.fft(sgmts_on[i], samples)
    # window func var.
    w_sgrOFF[i] = np.fft.fft(wOFF, samples)
    w_sgrON[i] = np.fft.fft(wON, samples)

    # for freq charactr.
    H += sgrON[i] / sgrOFF[i]
    # window func var.
    w_H += w_sgrON[i] / w_sgrOFF[i]


# frequency characteristic
H = np.abs(H) / N
# impulse response
IR = np.fft.ifft(H)

H = H[:int(samples/2)]
IR = IR[:int(samples/2)]

# window func var.
# frequency characteristic
w_H = np.abs(w_H) / N
# impulse response
w_IR = np.fft.ifft(w_H)

w_H = w_H[:int(samples/2)]
w_IR = w_IR[:int(samples/2)]


'''__________GENERATING OUTPUTS__________'''
# .wav audio files

dir = '../audio/'

# tone
out = ss.lfilter(H, [1], data_tone_off)
out /= np.abs(out).max()
outT = out.real
wav.write(dir + 'sim_maskon_tone.wav', fs, outT)

# tone w/ overlap-method
step = int(sgmts_size / 2)
N = int(data_tone_off.size / step) - 1
sgmts = split(data_tone_off, N, sgmts_size, step)
out = overlapAdd(sgmts, IR, N)
out /= np.abs(out).max()
outT_ova = out.real
wav.write(dir + 'sim_maskon_tone_overlap_add.wav', fs, outT_ova)

# tone w/ window function
out = ss.lfilter(w_H, [1], data_tone_off)
out /= np.abs(out).max()
outT_w = out.real
wav.write(dir + 'sim_maskon_tone_window.wav', fs, outT_w)


# tone final
step = int(sgmts_size / 2)
N = int(data_tone_off.size / step) - 1
sgmts = split(data_tone_off, N, sgmts_size, step)
out = overlapAdd(sgmts, w_IR, N)
out /= np.abs(out).max()
outT_f = out.real
wav.write(dir + 'sim_maskon_tone_final.wav', fs, outT_f)




sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
sentenceON, fs = soundfile.read('../audio/maskon_sentence.wav')

# sentence
out = ss.lfilter(H, [1], sentenceOFF)
out /= np.abs(out).max()
outS = out.real
wav.write(dir + 'sim_maskon_sentence.wav', fs, outS)

# sentence w/ overlap-method
step = int(sgmts_size / 2)
N = int(sentenceOFF.size / step) - 1
sgmts = split(sentenceOFF, N, sgmts_size, step)
out = overlapAdd(sgmts, IR, N)
out /= np.abs(out).max()
outS_ova = out.real
wav.write(dir + 'sim_maskon_sentence_overlap_add.wav', fs, outS_ova)

# sentence w/ window function
out = ss.lfilter(w_H, [1], sentenceOFF)
out /= np.abs(out).max()
outS_w = out.real
wav.write(dir + 'sim_maskon_sentence_window.wav', fs, outS_w)





# sentence final
step = int(sgmts_size / 2)
N = int(sentenceOFF.size / step) - 1
sgmts = split(sentenceOFF, N, sgmts_size, step)
out = overlapAdd(sgmts, w_IR, N)
out /= np.abs(out).max()
outS_f = out.real
wav.write(dir + 'sim_maskon_sentence_final.wav', fs, outS_f)




# sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
# step = int(sgmts_size / 2)
# N = int(sentenceOFF.size / step) - 1
# ramce = split(sentenceOFF, N, sgmts_size, step)
# out = overlapAdd(ramce, io, N)
# out /= np.abs(out).max()
# outS = out.real
# wav.write(dir + 'sentence.wav', fs, outS)


'''__________GENERATING OUTPUTS__________'''
# graphs

dir = '../img/'

# --------------------------    DATA
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(np.arange(data_tone_off.size), data_tone_off)
# ax2.plot(np.arange(data_tone_on.size), data_tone_on)
# --------------------------    TONES
fig, axs = plt.subplots(6)
fig.set_size_inches(10,10)

data_tone_off /= np.abs(data_tone_off).max()
data_tone_on /= np.abs(data_tone_on).max()

axs[0].set_title('Mask off')
axs[0].plot(np.arange(data_tone_off.size), data_tone_off, )
axs[1].set_title('Mask on')
axs[1].plot(np.arange(data_tone_on.size), data_tone_on)
axs[2].set_title('Sim (basic)')
axs[2].plot(np.arange(outT.size), outT)
axs[3].set_title('Sim (overlap-add)')
axs[3].plot(np.arange(outT_ova.size), outT_ova)
axs[4].set_title('Sim (window function)')
axs[4].plot(np.arange(outT_w.size), outT_w)


axs[5].set_title('Sim (final)')
axs[5].plot(np.arange(outT_f.size), outT_f)

plt.tight_layout()
plt.savefig(dir + 'tones.png')

# --------------------------    [3] SEGMENTS
x = np.arange(0., sgmts_size_s, sgmts_size_s / sgmts_size)
plt.figure(figsize=(10,4))
plt.plot(x, sgmts_off[50], label='mask off')
plt.plot(x, sgmts_on[50], label='mask on')

plt.xlabel('vzorky')
plt.suptitle('Rámce')
plt.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'ramce.png')


# --------------------------    [4] SEGMENT
fig, axs = plt.subplots(4)
fig.set_size_inches(10,7)

axs[0].plot(x, sgmts_off[20], label='mask off')
axs[0].set_title('Rámec')
axs[0].set_xlabel('čas')
# --------------------------    [4] CENTRAL CLIPPING
x = list(range(sgmts_size))
cc = central_clip(sgmts_off[20])
axs[1].plot(x, cc)
axs[1].set_title('Centrálne klipovanie so 70 %')
axs[1].set_xlabel('vzorky')
# --------------------------    [4] AUTOCORRELATION
ac = autocorr(cc)
lg = lag(ac, prah)
val = ac[lg]
axs[2].plot(x, ac)
axs[2].set_title('Autokorelácia')
axs[2].set_xlabel('vzorky')
axs[2].plot([prah, prah], [0, 20], 'black', lw=2, label='Prah')
axs[2].plot([lg, lg], [0, val], 'red', lw=2, label='Lag')
axs[2].legend()
# --------------------------    [4] BASE FREQs
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


# --------------------------    [5] SPECTROGRAMS
sgrOFF_log = np.transpose(10 * np.log10(np.abs(sgrOFF)**2))
sgrON_log = np.transpose(10 * np.log10(np.abs(sgrON)**2))


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


# --------------------------    [6] FREQ CHARACTERISTIC
plt.figure(figsize=(8,6))
plt.suptitle("Frekvenčná charakteristika rúšky")

plt.plot(np.arange(H.size) / samples * fs/2, H)
plt.xlabel('frekvencia')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'freqch.png')
# --------------------------    [7] IMPULSE RESPONSE
plt.figure(figsize=(8,6))
plt.suptitle("Impulzná odozva rúšky")
plt.plot(np.arange(IR.size), IR.real)
plt.xlabel('čas')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'impOdoz.png')


# --------------------------    [8] OUTCOME
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

sim = ss.lfilter(H, [1], sentenceOFF)
sim /= np.abs(sim).max()

sentenceOFF /= np.abs(sentenceOFF).max()
sentenceON /= np.abs(sentenceON).max()

axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF)
axs[0].set_xlabel('vzorky')

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)
axs[1].set_xlabel('vzorky')

axs[2].set_title('Veta so simulovanou rúškou')
axs[2].plot(np.arange(sim.size), sim)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_basic.png')


# ----------------------------------------------------
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

step = int(sgmts_size / 2)
N = int(sentenceOFF.size / step) - 1
ramce = split(sentenceOFF, N, sgmts_size, step)
tmp = overlapAdd(ramce, IR, N)
tmp /= np.abs(tmp).max()
sim = tmp.real

axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF)
axs[0].set_xlabel('vzorky')

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)
axs[1].set_xlabel('vzorky')

axs[2].set_title('Veta so simulovanou rúškou (metóda overlap-add)')
axs[2].plot(np.arange(sim.size), sim)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_overlap_add.png')


# ----------------------------------------------------
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

sim = ss.lfilter(w_H, [1], sentenceOFF)
sim /= np.abs(sim).max()

axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF)
axs[0].set_xlabel('vzorky')

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)
axs[1].set_xlabel('vzorky')

axs[2].set_title('Veta so simulovanou rúškou (okienková funkcia)')
axs[2].plot(np.arange(sim.size), sim)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_window.png')


# ----------------------------------------------------
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

step = int(sgmts_size / 2)
N = int(sentenceOFF.size / step) - 1
ramce = split(sentenceOFF, N, sgmts_size, step)
tmp = overlapAdd(ramce, w_IR, N)
tmp /= np.abs(tmp).max()
sim = tmp.real

axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF)
axs[0].set_xlabel('vzorky')

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)
axs[1].set_xlabel('vzorky')

axs[2].set_title('Veta so simulovanou rúškou (final)')
axs[2].plot(np.arange(sim.size), sim)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_final.png')


# ----------------------------------------------------
