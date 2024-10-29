import numpy as np
import matplotlib.pyplot as plt
import os

name = np.loadtxt(os.getcwd() + '/SN/Analysis/planetary_data.dat', usecols=0, unpack=True, dtype=str)
gmag = np.loadtxt(os.getcwd() + '/SN/Analysis/planetary_data.dat', usecols=8, unpack=True)

fp_thm, fp_refl3, fp_refl5 = np.loadtxt(os.getcwd() + '/SN/Analysis/Occ_depths.dat', usecols=(1,2,3), unpack=True)

plt.errorbar(gmag, fp_thm+fp_refl3, fmt='.')
plt.show()

print(name[fp_thm + fp_refl3 > 50.], (fp_thm + fp_refl3)[fp_thm + fp_refl3 > 50.], gmag[fp_thm + fp_refl3 > 50.])