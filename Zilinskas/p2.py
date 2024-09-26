import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table
import astropy.units as u
from glob import glob
import os

# CHEOPS response function
lam, cheops_pass = np.loadtxt(os.getcwd() + '/Data/cheops_response_fun.txt', usecols=(0,1), unpack=True)
lam = lam/1e4       # Converting to the microns

## Interpolating CHEOPS bandpass
wav_models = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/Emission/BSE/F00/star/55Cnce.dat', usecols=0, unpack=True)
# Computing TESS bandpass on the wavelengths of theoretical spectra
lam_pass, trans_fun = np.copy(wav_models), np.zeros(len(wav_models))
spln2 = interp1d(x=lam, y=cheops_pass)
for i in range(len(lam_pass)):
    if (lam_pass[i]<np.min(lam))or(lam_pass[i]>np.max(lam)):
        trans_fun[i] = 0.
    else:
        trans_fun[i] = spln2(lam_pass[i])

# Loading the planet properties table
star = np.loadtxt(os.getcwd() + '/Zilinskas/Analysis/gmag.dat', usecols=0, unpack=True, dtype=str)
gmag, subT = np.loadtxt(os.getcwd() + '/Zilinskas/Analysis/gmag.dat', usecols=(1,2), unpack=True)

# All available spectra
lst = glob(os.getcwd() + '/Data/LavaPlanets/Emission/BSE/F00/star/*.dat')
star_data, planet_data = np.array([]), np.array([])
for i in range(len(lst)):
    nm1 = lst[i].split('/')[-1][:-5]
    nm2 = lst[i].split('/')[-1][:-4]
    star_data = np.hstack((star_data, nm1))
    planet_data = np.hstack((planet_data, nm2))

# And now computing estimated depth for CHEOPS
occultation_dep = np.zeros(len(star_data))
gmag_data, subT_data = np.zeros(len(star_data)), np.zeros(len(star_data))
for i in range(len(star_data)):
    # First opening the file
    wav1, fpfs1 = np.loadtxt(os.getcwd() + '/Data/LavaPlanets/Emission/BSE/F00/star/' + planet_data[i] + '.dat', usecols=(0,1), unpack=True)
    
    ## Computing occultation depth
    occ1 = np.sum(fpfs1*trans_fun)/np.sum(trans_fun)
    occultation_dep[i] = occ1

    ## Anf now gmag, subT
    loc = np.where(star == star_data[i])
    if len(loc[0]) == 0:
        print(star_data[i])
        gmag_data[i] = 1000
        subT_data[i] = 1000
    else:
        gmag_data[i] = gmag[loc[0][0]]
        subT_data[i] = subT[loc[0][0]]

# And saving those data
f1 = open(os.getcwd() + '/Zilinskas/Analysis/data.dat', 'w')
f1.write('# Star\t\tPlanet\t\tGmag\t\tsubT\t\tOccDep\n')
for i in range(len(star_data)):
    f1.write(star_data[i] + '\t' + planet_data[i] + '\t' + str(gmag_data[i]) + '\t' + str(subT_data[i]) + '\t' + str(occultation_dep[i]) + '\n')
f1.close()