import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import matplotlib
import matplotlib.cm as cm
import os

# Defining colors from colormap (will define 10 colors -- and will choose 2 out of them)
chex = np.array([])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)
for i in range(11):
    c1 = cm.plasma(norm(100*i))     # Use cm.viridis for viridis colormap and so on...
    c2 = matplotlib.colors.rgb2hex(c1, keep_alpha=True)
    chex = np.hstack((chex, c2))

tess_expected_dep = 27.40

# TP profile
p, t = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/TP/BSE/F00/TOI-561b.txt', usecols=(0,1), unpack=True)
p1, t1 = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/TP/BSE/F80/TOI-561b.txt', usecols=(0,1), unpack=True)
## Converting pressure to bars
p = p * 1e-6
p1 = p1 * 1e-6

# And the emission spectrum
wav, fl = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/Emission/BSE/F00/star/TOI-561b.dat', usecols=(0,1), unpack=True)
wav1, fl1 = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/Emission/BSE/F80/star/TOI-561b.dat', usecols=(0,1), unpack=True)

# Bandpass
wav_band, trans_band = np.loadtxt(os.getcwd() + '/Data/tess_response_fun.txt', usecols=(0,1), unpack=True)
wav_band = wav_band * 1e6
trans_band = 50*(trans_band - np.min(trans_band)) - 50.

mean_band = (wav_band[-1]+wav_band[0])/2
error_band = wav_band[-1]-mean_band

# The figure
fig = plt.figure(figsize=(12,5.5))
gs = gd.GridSpec(1,2, width_ratios=[1,3], wspace=0.01)

ax1 = plt.subplot(gs[0])
ax1.plot(t, p, color=chex[5], label='Non-evolved surface')
ax1.plot(t1, p1, color=chex[1], label='Evolved surface')
ax1.invert_yaxis()
ax1.set_xticks(ticks=[3200, 3300, 3400, 3500, 3600, 3700, 3800], labels=['3200', '', '', '3500', '', '', '3800'])
ax1.set_yscale('log')
ax1.legend(loc='best')
ax1.set_xlabel(r'Temperature [K]', fontsize=16)
ax1.set_ylabel(r'Pressure [bar]', fontsize=16)

ax2 = plt.subplot(gs[1])
#axins = ax2.inset_axes([0.025, 0.60, 0.45, 0.35], visible=True, zorder=50)

ax2.plot(wav, fl, color=chex[5], zorder=1)
ax2.plot(wav1, fl1, color=chex[1], zorder=10, alpha=0.7)
#ax2.plot(wav_band, trans_band, color=chex[6])
ax2.errorbar(mean_band, tess_expected_dep, xerr=error_band, yerr=np.array([11.35, 10.87]).reshape((2,1)), fmt='*', ms=12, mfc='white', c='g', capsize=3, elinewidth=1.5, mew=1.5, zorder=100)
#ax2.text(1.2, -48, 'CHEOPS bandpass', fontsize=15, color=chex[6])
ax2.set_xscale('log')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_xlabel(r'Wavelength [$\mu$m]', fontsize=16)
ax2.set_ylabel(r'F$_p$/F$_\star$ [ppm]', fontsize=16)#, rotation=270, labelpad=20)

#ax2.indicate_inset_zoom(axins, edgecolor="black")
ax2.set_ylim([0, 180])


plt.setp(ax1.get_xticklabels(), fontsize=14)
plt.setp(ax2.get_xticklabels(), fontsize=14)
plt.setp(ax1.get_yticklabels(), fontsize=14)
plt.setp(ax2.get_yticklabels(), fontsize=14)

plt.tight_layout()

#plt.show()
plt.savefig(os.getcwd() + '/Zilinskas/Analysis/toi561tess.png', dpi=500)