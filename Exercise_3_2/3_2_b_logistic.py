import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

samples = 1000


# Parameters
r = 4
x0 = 0.1

# Iterations
iterations = 1000

def logistic(iter, r_val, x_start):
    x = x_start
    x_limit = 1.0 - 1.0/r_val
    values = []

    for i in range(iter):
        values.append(x)
        x = x*r_val*(1.0-x)

    deviation = x_limit - values[-1]

    return x_limit, deviation, values



x_limit_value, delta, x_sequence = logistic(iter=iterations,
                                            r_val=r,
                                            x_start=x0)

xn = x_sequence

def autocorr_cyc(time_series, N):
    autocorr = []
    for i in range(N):
        an = sum(time_series[j]*time_series[(j+i) % N] for j in range(N))
        autocorr.append(an)
    return autocorr

def autocorr(time_series, n):
    mean = np.mean(time_series)
    #var = np.sum([(i-mean)**2 for i in time_series])
    sum = 0
    var = 0
    for i in range(len(time_series)-n):
        sum += (time_series[i] - mean)*(time_series[i+n] - mean)
        #var += (time_series[i] - mean)**2
    return sum

def fft2(time_series):
    ts_ft_ampl = np.fft.fft(time_series)
    ts_ft_freq = np.fft.fftfreq(samples, d=1)
    ft_phases = np.angle(ts_ft_ampl)
    return ts_ft_freq[:samples//2], ts_ft_ampl[:samples//2], ft_phases[:samples//2]


xn_autocorr_cyc = autocorr_cyc(xn,samples)

xn_ft_freq, xn_ft_ampl, xn_ft_phases = fft2(xn)
cyc_ft_freq, cyc_ft_ampl, cyc_ft_phases = fft2(xn_autocorr_cyc)

fig,ax = plt.subplots(2,3)
ax[0,0].plot(np.arange(samples),xn, label='time-series [Xn]')
ax[0,1].plot(xn_ft_freq, np.abs(xn_ft_ampl)**2, label='Power Spectrum ([Xn])')
ax[0,2].plot(xn_ft_freq, xn_ft_phases, label='Fourier-Phase ([Xn])')
ax[1,0].plot(np.arange(samples), xn_autocorr_cyc, label='Autocorrelation [An]')
ax[1,1].plot(cyc_ft_freq, cyc_ft_ampl,  label='Fourier-Coeff. ([An])')
ax[1,2].plot(cyc_ft_freq, cyc_ft_phases, label='Fourier-Phase ([An])')
for ax in ax.flat:
    ax.legend()  # Add a legend to each subplot
# fig,ax = plt.subplots(1,2)
# ax[0].plot(xn_ft_freq, np.abs(xn_ft_ampl)**2, label='Power Spectrum ([Xn])')
# ax[1].plot(cyc_ft_freq, cyc_ft_ampl,  label='Fourier-Coeff. ([An])')
# for ax in ax.flat:
#     ax.legend()  # Add a legend to each subplot

plt.show()