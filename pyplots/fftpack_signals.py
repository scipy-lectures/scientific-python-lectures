import numpy as np
from scipy import fftpack
import pylab as pl

time_step = 0.1
period = 5.

time_vec = np.arange(0, 20, time_step)
sig = np.sin(2 * np.pi / period * time_vec) + np.cos(10 * np.pi * time_vec)

sample_freq = fftpack.fftfreq(sig.size, d=time_step)
sig_fft = fftpack.fft(sig)
pidxs = np.where(sample_freq > 0)
freqs, power = sample_freq[pidxs], np.abs(sig_fft)[pidxs]
freq = freqs[power.argmax()]

sig_fft[np.abs(sample_freq) > freq] = 0
main_sig = fftpack.ifft(sig_fft)

pl.figure()
pl.plot(time_vec, sig)
pl.plot(time_vec, main_sig, linewidth=3)
pl.ylabel('Amplitude')
pl.xlabel('Time [s]')


