import numpy as np
import matplotlib.pyplot as plt

samples = 1000

np.random.seed(42)
#xn = np.random.normal(0,1,samples)
xn = np.sin(np.linspace(0, 10*np.pi, samples))

def autocorr_cyc(time_series, N):
    autocorr = []
    for i in range(N):
        an = sum(time_series[j]*time_series[(j+i) % N] for j in range(N))
        autocorr.append(an/samples)
    return autocorr


def fft2(time_series):
    ts_ft_ampl = np.fft.fft(time_series)/samples
    ts_ft_freq = np.fft.fftfreq(samples, d=1)
    ft_phases = np.angle(ts_ft_ampl)
    return ts_ft_freq[:samples//2], ts_ft_ampl[:samples//2], ft_phases[:samples//2]


# Verification for the cyclically defined autocorrelation
xn_autocorr_cyc = autocorr_cyc(xn,samples)
cyc_xn_ft_freq, cyc_xn_ft_ampl, cyc_xn_ft_phases = fft2(xn)
cyc_ft_freq, cyc_ft_ampl, cyc_ft_phases = fft2(xn_autocorr_cyc)

fig,ax = plt.subplots(1,1)
ax.plot(np.arange(samples), xn_autocorr_cyc, label='Autocorrelation [An]')
pi_ticks = np.arange(0, 11) * np.pi  # 0 to 10π
pi_labels = [r"$%d\pi$" % i if i != 0 else r"$0$" for i in range(11)]
plt.xticks(ticks=pi_ticks, labels=pi_labels)
ax.set_xlabel(r'$\tau$', fontsize=20)
ax.set_ylabel(r'$A_{X_i}(\tau)$', fontsize=20)
plt.grid()
plt.title(r'Autocorrelation of the time-series $X_i = Sin(x)$', fontsize=20)
plt.legend(fontsize=20, loc='upper right')
plt.show()

# fig,ax = plt.subplots(2,3)
# plt.title('WK-Theorem for cyclically defined autocorrelation')
# ax[0,0].plot(np.arange(samples),xn, label='time-series [Xn]')
# ax[0,1].plot(cyc_xn_ft_freq, np.abs(cyc_xn_ft_ampl)**2, label='Power Spectrum ([Xn])')
# ax[0,2].plot(cyc_xn_ft_freq, cyc_xn_ft_phases, label='Fourier-Phase ([Xn])')
# ax[1,0].plot(np.arange(samples), xn_autocorr_cyc, label='Autocorrelation [An]')
# ax[1,1].plot(cyc_ft_freq, cyc_ft_ampl,  label='Fourier-Coeff. ([An])')
# ax[1,2].plot(cyc_ft_freq, cyc_ft_phases, label='Fourier-Phase ([An])')
# for ax in ax.flat:
#     ax.legend()  # Add a legend to each subplot
# plt.show()


# Verification for the non-cyclically defined autocorrelation

# def autocorr(time_series, N):
#     autocorr = []
#     for i in range(N):
#         an = sum(time_series[j]*time_series[(j+i)] for j in range(N-i))
#         autocorr.append(an)
#     return autocorr

# xn_autocorr = autocorr(xn,samples)
# xn_ft_freq, xn_ft_ampl, xn_ft_phases = fft2(xn)
# ft_freq, ft_ampl, ft_phases = fft2(xn_autocorr)
#
# fig,ax = plt.subplots(2,3)
# plt.title('WK-Theorem for non-cyclically defined autocorrelation')
# ax[0,0].plot(np.arange(samples),xn, label='time-series [Xn]')
# ax[0,1].plot(xn_ft_freq, np.abs(xn_ft_ampl)**2, label='Power Spectrum ([Xn])')
# ax[0,2].plot(xn_ft_freq, xn_ft_phases, label='Fourier-Phase ([Xn])')
# ax[1,0].plot(np.arange(samples), xn_autocorr, label='Autocorrelation [An]')
# ax[1,1].plot(ft_freq, ft_ampl,  label='Fourier-Coeff. ([An])')
# ax[1,2].plot(ft_freq, ft_phases, label='Fourier-Phase ([An])')
# for ax in ax.flat:
#     ax.legend()  # Add a legend to each subplot
# plt.show()