import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import os

star = np.loadtxt(os.getcwd() + '/Zilinskas/Analysis/data.dat', usecols=0, unpack=True, dtype=str)
gmag, subT, occ = np.loadtxt(os.getcwd() + '/Zilinskas/Analysis/data.dat', usecols=(2,3,4), unpack=True)

#teq_lbls = np.arange(0,3500,500)

# Figure
fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.scatter(gmag, occ, c=subT, s=0)

norm = matplotlib.colors.Normalize(vmin=1000., vmax=4000., clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='plasma')
vcolor = np.array([(mapper.to_rgba(v)) for v in subT])

for x, y, clr in zip(gmag, occ, vcolor):
        axs.errorbar(x, y, fmt='o', c=clr, mew=1.5, zorder=10)

mapper.set_array([])
cbar = plt.colorbar(mapper)
cbar.set_label('Substellar Temperature', fontsize=20, rotation=270, labelpad=20)
#cbar.set_ticklabels(fontsize=16, ticklabels=teq_lbls)

axs.set_xlabel("Gaia magnitude", fontsize=20)
axs.set_ylabel("Expected CHEOPS Occultation depth", fontsize=20)
#axs.set_xlim([0.1, 2.])
#axs.set_ylim([0.4, 20.])

axs.set_xscale('log')
axs.set_yscale('log')

plt.xticks(fontsize=16);
plt.yticks(fontsize=16);

plt.tight_layout()

plt.show()
#plt.savefig('population_usps.pdf')

loc2 = np.where(occ > 20.)[0]
for i in range(len(loc2)):
    print('Star: ' + star[loc2[i]] + '; Gmag: ' + str(gmag[loc2[i]]) + '; Occ: ' + str(occ[loc2[i]]))