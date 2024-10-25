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

cheops_expected_dep = 39.696787188657325

# TP profile
p, t = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/TP/BSE/F00/TOI-1807b.txt', usecols=(0,1), unpack=True)
## Converting pressure to bars
p = p * 1e-6

# And the emission spectrum
wav, fl = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/Emission/BSE/F00/star/TOI-1807b.dat', usecols=(0,1), unpack=True)

# Bandpass
wav_band, trans_band = np.loadtxt(os.getcwd() + '/Data/cheops_response_fun.txt', usecols=(0,1), unpack=True)
wav_band = wav_band / 1e4
trans_band = 50*(trans_band - np.min(trans_band)) - 50.

mean_band = (wav_band[-1]+wav_band[0])/2
error_band = wav_band[-1]-mean_band

# The figure
fig = plt.figure(figsize=(12,5.5))
gs = gd.GridSpec(1,2, width_ratios=[1,3], wspace=0.01)

ax1 = plt.subplot(gs[0])
ax1.plot(t, p, color=chex[1])
ax1.invert_yaxis()
ax1.set_xticks(ticks=[2900, 2950, 3000, 3050, 3100, 3150, 3200], labels=['', '2950', '', '', '', '3150', ''])
ax1.set_yscale('log')
ax1.set_xlabel(r'Temperature [K]', fontsize=16)
ax1.set_ylabel(r'Pressure [bar]', fontsize=16)

ax2 = plt.subplot(gs[1])
axins = ax2.inset_axes([0.025, 0.60, 0.45, 0.35], visible=True, zorder=50)

ax2.plot(wav, fl, color=chex[1])
ax2.plot(wav_band, trans_band, color=chex[6])
ax2.errorbar(mean_band, cheops_expected_dep, fmt='*', ms=12, mfc='white', c=chex[8], capsize=3, elinewidth=1.5, mew=1.5, zorder=100)
ax2.text(1.2, -48, 'CHEOPS bandpass', fontsize=15, color=chex[6])
ax2.set_xscale('log')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_xlabel(r'Wavelength [$\mu$m]', fontsize=16)
ax2.set_ylabel(r'F$_p$/F$_\star$ [ppm]', fontsize=16)#, rotation=270, labelpad=20)

axins.plot(wav, fl, color=chex[1])
axins.errorbar(mean_band, cheops_expected_dep, xerr=error_band, fmt='*', ms=12, mfc='white', c=chex[8], capsize=3, elinewidth=1.5, mew=1.5, zorder=100)
axins.yaxis.set_label_position("right")
axins.yaxis.tick_right()
axins.set_xscale('log')
axins.set_xticks(ticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=['0.4', '', '0.6', '', '0.8', '', '1.0'])
axins.set_xlim([3300.0e-4, 11000.0e-4])
axins.set_ylim([0, 110])

ax2.indicate_inset_zoom(axins, edgecolor="black")
ax2.set_ylim([-50, 500])


plt.setp(ax1.get_xticklabels(), fontsize=14)
plt.setp(ax2.get_xticklabels(), fontsize=14)
plt.setp(ax1.get_yticklabels(), fontsize=14)
plt.setp(ax2.get_yticklabels(), fontsize=14)

plt.tight_layout()

#plt.show()
plt.savefig(os.getcwd() + '/Zilinskas/Analysis/toi1807.png', dpi=500)