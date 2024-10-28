import numpy as np
from utils import planck_func
from scipy.integrate import simpson
import astropy.units as u
import os

# Reading the data
name = np.loadtxt(os.getcwd() + '/SN/Analysis/planetary_data.dat', usecols=0, unpack=True, dtype=str)
rprs, ar, teq, teff = np.loadtxt(os.getcwd() + '/SN/Analysis/planetary_data.dat', usecols=(2,3,6,7), unpack=True)

# CHEOPS transmission function
wav, trans = np.loadtxt(os.getcwd() + '/Data/cheops_response_fun.txt', usecols=(0,1), unpack=True)
wav = (wav * u.AA).to(u.micron)

fpfs_thm, fpfs_refl3, fpfs_refl5 = np.zeros(len(name)), np.zeros(len(name)), np.zeros(len(name))

for i in range(len(name)):
    # Thermal component
    planck_planet = planck_func(lam = wav, temp = teq[i] * u.K).value
    planck_star = planck_func(lam = wav, temp = teff[i] * u.K).value
    fp1 = simpson(planck_planet * trans, wav.value)/simpson(planck_star * trans, wav.value)
    fpfs_thm[i] = rprs[i] * fp1 * 1e6
    # Reflective component
    fpfs_refl3[i], fpfs_refl5[i] = 1e6 * 0.3 * ((rprs[i] / ar[i])**2), 1e6 * 0.5 * ((rprs[i] / ar[i])**2)

# Saving the data of occultation depths
f1 = open(os.getcwd() + '/SN/Analysis/Occ_depths.dat', 'w')
f1.write('#Name\tFpThm\tFpRefl3\tFpRefl5\n')
for i in range(len(name)):
    f1.write(str(name[i]) + '\t' + str(fpfs_thm[i]) + '\t' + str(fpfs_refl3[i]) + '\t' + str(fpfs_refl5[i]) + '\n')
f1.close()