import soundfile
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as ss
import scipy.io.wavfile as wav
from scipy.stats import pearsonr

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

def dft(x_in, n):
    x = np.zeros(n, dtype=complex)
    x[:x_in.size] = x_in
    X = np.zeros(n, dtype=complex)

    X = [sum((x[k] * np.exp(np.complex(0, -2 * math.pi * i * k / n))
            for k in range(n)))
                for i in range(n)]
    return X

def idft(x):
    N = x.size
    K = x[0].size
    X = np.zeros(N, dtype=complex)

    X[:-1] = [sum((x[k] * np.exp(np.complex(0, 2 * math.pi * k * N / n))
            for k in range(K)))
                for n in range(1, N)]

    return X / N

def overlapAdd(x, h, N):
    M = 1024
    size = x[0].size
    step = int(size / 2)
    out = np.zeros(step * N + M, dtype=complex)
    H = np.fft.fft(h, M)

    for i in range(0,N):
        pos = step * i
        ramec = np.zeros(M, dtype=complex)
        ramec[:size] = x[i]
        f = np.fft.ifft(np.fft.fft(ramec, M) * H)
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

def median_filter(l, i, n):
    if ((i-2 < 0) or (i+2) >= n):
        return l[i]
    else:
        med = np.median([l[i-2], l[i-1], l[i], l[i+1], l[i+2]])
        if ((med < 2*l[i]) and (med > l[i] / 2)):
            return l[i]
        else:
            return med




# get records
data_tone_off, fs = soundfile.read('../audio/maskoff_tone.wav')
data_tone_on, fs = soundfile.read('../audio/maskon_tone.wav')
# trim to actual sound
data_tone_off = data_tone_off[13200:63300]
data_tone_on = data_tone_on[11400:60600]


# startOFF = 0
# startON = 0
# size = 16000
# max_ = 0
# for j in range(8000, 30000, 40):
#     for i in range(10000, data_tone_off.size - size, 10):
#         c = np.corrcoef(data_tone_on[j:j+size], data_tone_off[i:i+size])
#
#         if (c[0,1] > max_):
#             max_ = c[0,1]
#             startOFF = i
#             startON = j
#
# print(startOFF, startON)
# quit()

startOFF = 23040
startON = 14240


# startOFF = 20000
tone_off = data_tone_off[startOFF:startOFF+fs]
t1 = np.arange(tone_off.size) / fs

# startON = 21000
tone_on = data_tone_on[startON:startON+fs]
t2 = np.arange(tone_on.size) / fs

t1 -= np.mean(t1)
t2 -= np.mean(t2)

t1 /= np.abs(t1).max()
t2 /= np.abs(t2).max()


'''__________SEGMENTS__________'''
N = 99
sgmt_size_s = 0.02
sgmt_size = int(sgmt_size_s * fs)
step = int(0.01 * fs)

sgmts_off = split(tone_off, N, sgmt_size, step)
sgmts_on = split(tone_on, N, sgmt_size, step)



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
# only_match var.
om_H = np.zeros(samples, dtype=complex)
wom_H = np.zeros(samples, dtype=complex)

# get array of L for each segment
L_off = np.zeros(N)
L_on = np.zeros(N)
for i in range(0,N):
    # mask off
    cc = central_clip(sgmts_off[i])
    ac = autocorr(cc)
    L_off[i] = lag(ac, prah)

    # mask on
    cc = central_clip(sgmts_on[i])
    ac = autocorr(cc)
    L_on[i] = lag(ac, prah)

# get base freq of each segment
for i in range(0,N):
    base_freqsOFF[i] = fs / median_filter(L_off[:], i, N)
    base_freqsON[i] = fs / median_filter(L_on[:], i, N)

cnt = 0
for i in range(0, N):
    # apply window function
    wOFF = sgmts_off[i] * np.hamming(sgmt_size)
    wON = sgmts_on[i] * np.hamming(sgmt_size)

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
    # only_match var.
    if (np.isclose(base_freqsOFF[i], base_freqsON[i])):
        om_H += sgrON[i] / sgrOFF[i]
        wom_H += w_sgrON[i] / w_sgrOFF[i]
        cnt += 1

# frequency characteristic
H = np.real(H) / N + 1j * np.imag(H) / N
# impulse response
IR = np.fft.ifft(H)

H = H[:int(samples/2)]
IR = IR[:int(samples/2)]


# window func var.
# frequency characteristic
w_H = np.real(w_H) / N + 1j * np.imag(w_H) / N
# impulse response
w_IR = np.fft.ifft(w_H)

w_H = w_H[:int(samples/2)]
w_IR = w_IR[:int(samples/2)]


# only_match var.
# frequency characteristic
om_H = np.real(om_H) / cnt + 1j * np.imag(om_H) / cnt
# impulse response
om_IR = np.fft.ifft(om_H)

om_H = om_H[:int(samples/2)]
om_IR = om_IR[:int(samples/2)]


# final
# frequency characteristic
f_H = np.real(wom_H) / cnt + 1j * np.imag(wom_H) / cnt
# impulse response
f_IR = np.fft.ifft(f_H)

f_H = f_H[:int(samples/2)]
f_IR = f_IR[:int(samples/2)]


'''__________GENERATING OUTPUTS__________'''
# .wav audio files

dir = '../audio/'

# tone
out = ss.lfilter(IR, [1], data_tone_off)
out /= np.abs(out).max()
outT = out.real
wav.write(dir + 'sim_maskon_tone.wav', fs, outT)

# tone w/ overlap-method
step = int(sgmt_size / 2)
N = int(data_tone_off.size / step) - 1
sgmts = split(data_tone_off, N, sgmt_size, step)
out = overlapAdd(sgmts, IR, N)
out /= np.abs(out).max()
outT_ova = out.real
wav.write(dir + 'sim_maskon_tone_overlap_add.wav', fs, outT_ova)

# tone w/ window function
out = ss.lfilter(w_IR, [1], data_tone_off)
out /= np.abs(out).max()
outT_w = out.real
wav.write(dir + 'sim_maskon_tone_window.wav', fs, outT_w)

# tone w/ only match
out = ss.lfilter(om_IR, [1], data_tone_off)
out /= np.abs(out).max()
outT_om = out.real
wav.write(dir + 'sim_maskon_tone_only_match.wav', fs, outT_om)



# tone final
step = int(sgmt_size / 2)
N = int(data_tone_off.size / step) - 1
sgmts = split(data_tone_off, N, sgmt_size, step)
out = overlapAdd(sgmts, f_IR, N)
out /= np.abs(out).max()
outT_f = out.real
wav.write(dir + 'sim_maskon_tone_final.wav', fs, outT_f)






sentenceOFF, fs = soundfile.read('../audio/maskoff_sentence.wav')
sentenceON, fs = soundfile.read('../audio/maskon_sentence.wav')

# sentence
out = ss.lfilter(IR, [1], sentenceOFF)
out /= np.abs(out).max()
outS = out.real
wav.write(dir + 'sim_maskon_sentence.wav', fs, outS)

# sentence w/ overlap-method
step = int(sgmt_size / 2)
N = int(sentenceOFF.size / step) - 1
sgmts = split(sentenceOFF, N, sgmt_size, step)
out = overlapAdd(sgmts, IR, N)
out /= np.abs(out).max()
outS_ova = out.real
wav.write(dir + 'sim_maskon_sentence_overlap_add.wav', fs, outS_ova)

# sentence w/ window function
out = ss.lfilter(w_IR, [1], sentenceOFF)
out /= np.abs(out).max()
outS_w = out.real
wav.write(dir + 'sim_maskon_sentence_window.wav', fs, outS_w)

# sentence w/ only match
out = ss.lfilter(om_IR, [1], sentenceOFF)
out /= np.abs(out).max()
outS_om = out.real
wav.write(dir + 'sim_maskon_sentence_only_match.wav', fs, outS_om)




# sentence final
step = int(sgmt_size / 2)
N = int(sentenceOFF.size / step) - 1
sgmts = split(sentenceOFF, N, sgmt_size, step)
out = overlapAdd(sgmts, f_IR, N)
out /= np.abs(out).max()
outS_f = out.real
wav.write(dir + 'sim_maskon_sentence_final.wav', fs, outS_f)




'''__________GENERATING OUTPUTS__________'''
# graphs

dir = '../img/'

# --------------------------    DATA
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.arange(data_tone_off.size), data_tone_off)
ax2.plot(np.arange(data_tone_on.size), data_tone_on)
plt.savefig(dir + 'data.png')


# --------------------------    TONES
fig, axs = plt.subplots(7)
fig.set_size_inches(10,10)

data_tone_off /= np.abs(data_tone_off).max()
data_tone_on /= np.abs(data_tone_on).max()

axs[0].set_title('Mask off')
axs[0].plot(np.arange(data_tone_off.size), data_tone_off, )
axs[1].set_title('Mask on')
axs[1].plot(np.arange(data_tone_on.size), data_tone_on)
axs[2].set_title('Sim (final)')
axs[2].plot(np.arange(outT_f.size), outT_f)

axs[3].set_title('Sim (basic)')
axs[3].plot(np.arange(outT.size), outT)
axs[4].set_title('Sim (overlap-add)')
axs[4].plot(np.arange(outT_ova.size), outT_ova)
axs[5].set_title('Sim (window function)')
axs[5].plot(np.arange(outT_w.size), outT_w)

out = ss.lfilter(om_IR, [1], data_tone_off)
out /= np.abs(out).max()
outT = out.real
axs[6].set_title('Sim (only match)')
axs[6].plot(np.arange(outT_w.size), outT)

plt.tight_layout()
plt.savefig(dir + 'tones.png')



# --------------------------    TONES
fig, axs = plt.subplots(3)

axs[0].set_title('Tón - Mask off')
axs[0].plot(np.arange(data_tone_off.size), data_tone_off, )
axs[1].set_title('Tón - Mask on')
axs[1].plot(np.arange(data_tone_on.size), data_tone_on)
axs[2].set_title('Tón - simulovaná rúška (final)')
axs[2].plot(np.arange(outT_f.size), outT_f)

plt.tight_layout()
plt.savefig(dir + 'finalTone.png')


# --------------------------    [3] SEGMENTS
x = np.arange(0., sgmt_size_s, sgmt_size_s / sgmt_size)
plt.figure(figsize=(10,4))
plt.plot(x, sgmts_off[20], label='mask off')
plt.plot(x, sgmts_on[20], label='mask on')

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
x = list(range(sgmt_size))
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
axs[3].set_ylabel('f0')
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
plt.figure(figsize=(8,4))
plt.suptitle("Frekvenčná charakteristika rúšky")

plt.plot((np.arange(H.size) / H.size) * fs/2, H.real)
plt.xlabel('frekvencia')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'freqch.png')
# --------------------------    [7] IMPULSE RESPONSE
plt.figure(figsize=(8,4))
plt.suptitle("Impulzná odozva rúšky")
plt.plot(np.arange(IR.size) / IR.size, IR.real)
plt.xlabel('čas')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'impOdoz.png')


# --------------------------    [8] OUTCOME
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

sim = ss.lfilter(IR, [1], sentenceOFF)
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
axs[2].plot(np.arange(sim.size), sim.real)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_basic.png')


# ----------------------------------------------------
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

step = int(sgmt_size / 2)
N = int(sentenceOFF.size / step) - 1
ramce = split(sentenceOFF, N, sgmt_size, step)
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
axs[2].plot(np.arange(sim.size), sim.real)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_overlap_add.png')


# ---------------------------------------------------- WINDOW FUNC
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

sim = ss.lfilter(w_IR, [1], sentenceOFF)
sim /= np.abs(sim).max()

axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF)
axs[0].set_xlabel('vzorky')

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)
axs[1].set_xlabel('vzorky')

axs[2].set_title('Veta so simulovanou rúškou (okienková funkcia)')
axs[2].plot(np.arange(sim.size), sim.real)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'out_window.png')


# hamming func
fig, (ax1, ax2) = plt.subplots(2)
window = np.hamming(51)
ax1.set_title('Hamming window')
ax1.set_xlabel('vzorky')
ax1.set_ylabel('amplitúda')
ax1.plot(window)

tmp = np.fft.fft(window, 1042)
freq = np.linspace(-0.5, 0.5, len(tmp))
mg = np.abs(np.fft.fftshift(tmp))
resp = 20 * np.log10(mg)
resp = np.clip(resp, -100, 100)
ax2.set_title('Odozva hammingovej okienkovej funkcie')
ax2.set_xlabel('frekvencia')
ax2.set_ylabel('magnitúda')
ax2.plot(freq, resp)

plt.tight_layout()
plt.savefig(dir + 'hamm.png')


# spectrs
seg = sgmts_off[20]
segw = seg * np.hamming(sgmt_size)
sgr = np.fft.fft(seg, 1024)
sgrw = np.fft.fft(segw, 1024)

sgr = sgr[:200]
sgrw = sgrw[:200]

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(np.arange(sgr.size), sgr.real)
ax1.set_title('Rámec bez okienkovej funkcie')
ax1.set_xlabel('vzorky')
ax2.set_title('Rámec s okienkovou funkciou')
ax2.plot(np.arange(sgrw.size), sgrw.real)
ax2.set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'spectrs.png')


# ----------------------------------------------------
fig, axs = plt.subplots(3)
fig.set_size_inches(10, 10)

axs[0].set_title('Veta bez rúšky')
axs[0].plot(np.arange(sentenceOFF.size), sentenceOFF)
axs[0].set_xlabel('vzorky')

axs[1].set_title('Veta s rúškou')
axs[1].plot(np.arange(sentenceON.size), sentenceON)
axs[1].set_xlabel('vzorky')

axs[2].set_title('Veta so simulovanou rúškou (final)')
axs[2].plot(np.arange(outS_f.size), outS_f.real)
axs[2].set_xlabel('vzorky')

plt.tight_layout()
plt.savefig(dir + 'finalSentence.png')


# ------------------------------------------ [12] LAG MEDIAN FILTER
# get faulty lag of segment
n = 50
prah = 0
cc = central_clip(sgmts_off[n])
ac = autocorr(cc)
L = lag(ac, prah)
val = ac[L]

x = list(range(sgmt_size))
plt.figure(figsize=(6,3))
plt.plot(x, ac)
plt.suptitle('N-násobný lag')
plt.xlabel('vzorky')
plt.plot([prah, prah], [-5, 20], 'black', lw=2, label='Prah')
plt.plot([L, L], [0, val], 'red', lw=2, label='Lag')
plt.plot(L, val, 'bo')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'lag_fix.png')


# ------------------------------------------
fig, (ax1,ax2) = plt.subplots(2)

ax1.set_title("Frekvenčná charakteristika rúšky (základná)")
ax1.plot(np.arange(H.size) / samples * fs/2, H.real)
ax1.set_xlabel('frekvencia')

ax2.set_title("Frekvenčná charakteristika rúšky (match only)")
ax2.plot(np.arange(om_H.size) / samples * fs/2, om_H.real)
ax2.set_xlabel('frekvencia')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(dir + 'matchonly.png')


# ------------------------------------------
