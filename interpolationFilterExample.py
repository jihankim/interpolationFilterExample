# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 19:49:20 2016

Upsampling using zero-padding and interpolation filter example

@author: Jihan Kim
"""

import numpy as np
import scipy.signal as signal
from matplotlib.pylab import *


# Magnitude Spectrum Calculation Fuction
def mag(x,n=1024):
    m = np.fft.fftshift( np.fft.fft(x, n))
    return 20*np.log10(abs(m))
    #return abs(m)




fs = 100; # Original Sampling Rate
ratio = 4; # Upsampling Ratio

#%% Generating Original Signal
f0 =  0.74; # Signal Frequency
s0 = np.sin(2*np.pi * f0/fs * np.arange(20)) # Orignal Signal



#%% Zero-Padding
sz = np.zeros(len(s0)*ratio)
for idx in range(len(s0)):
    sz[idx * ratio] = s0[idx]
    
    
    
#%% Low-pass Filter Design
N=16 # Filter Order
Wn = 0.125
b = signal.firwin(N, Wn, nyq=0.5, window = 'hamming')
w, h = signal.freqz(np.ones(4)/4)
figure(2);
clf();
plot(w/pi/2, 20*np.log10(abs(h)));
grid()
ylabel('[dB]')
ylim([-100, 0])

#%% Moving average Filter
maf = np.ones(ratio)/ratio
    
#%% Low-Pass Filtered Signal
#szf = np.convolve(sz, b,'same')
szf = np.convolve(sz, maf,'same') # using moving average filter


#%% Power Compensation
szfc = szf * ratio


#%% Plotting
figure(1);
clf();

# Displaying Original Signal
subplot(4,2,1);
stem(s0)
ylim([0, 1.5])
title('Orignal Signal (Time Domain)')

subplot(4,2,2);
plot(np.arange(-1024/2, 1024/2)/1024,  mag(s0))
xlim([-0.5, 0.5])
title('Orignal Signal (Frequency Domain)')
ylabel('[dB]')
ylim([-20, 20])

# Displaying Zero-Padded Signal
subplot(4,2,3);
stem(sz);
ylim([0, 1.5])
title('Zero-Padded Signal (Time Domain)')

subplot(4,2,4);
plot(np.arange(-1024/2, 1024/2)/1024,  mag(sz))
xlim([-0.5, 0.5])
title('Zero-Padded Signal (Frequency Domain)')
ylabel('[dB]')
ylim([-20, 20])

# Displaying Filtered Signal
subplot(4,2,5);
stem(szf);
ylim([0, 1.5])
title('Filtered Signal (Time Domain)')

subplot(4,2,6);
plot(np.arange(-1024/2, 1024/2)/1024,  mag(szf))
xlim([-0.5, 0.5])
title('Filtered Signal (Frequency Domain)')
ylabel('[dB]')
ylim([-20, 20])


# Displaying Filtered Signal
subplot(4,2,7);
stem(szfc);
ylim([0, 1.5])
title('Filtered Signal (Time Domain)')

subplot(4,2,8);
plot(np.arange(-1024/2, 1024/2)/1024,  mag(szfc))
xlim([-0.5, 0.5])
title('Filtered Signal (Frequency Domain)')
ylabel('[dB]')
#ylim([-20, 20])

tight_layout()