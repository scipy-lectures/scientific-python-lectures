
import numpy as np
from scipy import fftpack
import pylab as P

time_step = 0.1
period = 5.

time_vec = np.arange(0, 20, time_step)
sig = np.sin(2 * np.pi / period * time_vec) + np.cos(10 * np.pi * time_vec)

sample_freq = fftpack.fftfreq(sig.size, d=time_step)
sig_fft = fftpack.fft(sig)
pidxs = np.where(sample_freq > 0)
freqs, power = sample_freq[pidxs], np.abs(sig_fft)[pidxs]
freq = freqs[power.argmax()]

P.figure()
P.plot(freqs, power)
P.ylabel('Power')
P.xlabel('Frequency [Hz]')
axes = P.axes([0.3, 0.3, 0.5, 0.5])
P.title('Peak frequency')
P.plot(freqs[:8], power[:8])
P.setp(axes, yticks=[])
P.savefig('source/fftpack-frequency.png')

sig_fft[np.abs(sample_freq) > freq] = 0
main_sig = fftpack.ifft(sig_fft)

import pylab as P
P.figure()
P.plot(time_vec, sig)
P.plot(time_vec, main_sig, linewidth=3)
P.ylabel('Amplitude')
P.xlabel('Time [s]')
P.savefig('source/fftpack-signals.png')

