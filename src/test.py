import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk


maskoff_tone, fs1 = sf.read("../audio/maskoff_tone.wav")
maskon_tone, fs2 = sf.read("../audio/maskon_tone.wav")

t1 = np.arange(maskoff_tone.size) / fs1
t2 = np.arange(maskon_tone.size) / fs2

maskoff_tone = maskoff_tone / 2**15
maskon_tone = maskon_tone / 2**15
print (maskoff_tone.min(), maskoff_tone.max())
print (maskon_tone.min(), maskon_tone.max())

odkud = 0     # začátek segmentu v sekundách
kolik = 0.002  # délka segmentu v sekundách
odkud_vzorky = int(odkud * fs1)         # začátek segmentu ve vzorcích
pokud_vzorky = int((odkud+kolik) * fs1) # konec segmentu ve vzorcích

print(odkud_vzorky)
print(pokud_vzorky)
print(maskoff_tone.size)

s_seg = maskoff_tone[odkud_vzorky:pokud_vzorky]
N = s_seg.size

print(N)

# s_seg_spec = np.fft.fft(s_seg)
# G = 10 * np.log10(1/N * np.abs(s_seg_spec)**2)
#
#
# _, ax = plt.subplots(2,1)
#
# # np.arange(n) vytváří pole 0..n-1 podobně jako obyč Pythonovský range
# ax[0].plot(np.arange(s_seg.size) / fs + odkud, s_seg)
# ax[0].set_xlabel('$t[s]$')
# ax[0].set_title('Segment signalu $s$')
# ax[0].grid(alpha=0.5, linestyle='--')
#
# f = np.arange(G.size) / N * fs
# # zobrazujeme prvni pulku spektra
# ax[1].plot(f[:f.size//2+1], G[:G.size//2+1])
# ax[1].set_xlabel('$f[Hz]$')
# ax[1].set_title('Spektralni hustota vykonu [dB]')
# ax[1].grid(alpha=0.5, linestyle='--')
#
# plt.tight_layout()
#
#
# plt.figure(figsize=(6,3))
# plt.plot(t1, maskoff_tone)
# # plt.plot(t2, maskon_tone)
#
# plt.gca().set_xlabel('$t[s]$')
# plt.gca().set_title('Zvukový signál')
#
# plt.tight_layout()
# plt.show()
