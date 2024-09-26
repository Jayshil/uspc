import numpy as np
from astropy.table import Table
import os

# This file is to find gmag for all targets

## First, let's get the planet list (with necessary properties)
tab = Table.read(os.getcwd() + '/Data/LavaPlanets/PlanetList/LavaPlanetsConfirmed1.csv', delimiter=',')
tab_gmag = Table.read(os.getcwd() + '/Data/PSCompPars_2024.09.26_07.11.26.votable')

stars, subT = np.asarray(tab['st_name']), np.asarray(tab['pl_subT'])
gmag = np.zeros(len(stars))

tab_gmag_st, tab_gmag_gmag = np.asarray(tab_gmag['hostname']), np.asarray(tab_gmag['sy_gaiamag'])
for i in range(len(tab_gmag_st)):
    tab_gmag_st[i] = tab_gmag_st[i].replace(' ', '')

for i in range(len(stars)):
    st1 = np.where(tab_gmag_st == stars[i])
    if len(st1[0]) == 0:
        print(stars[i])
        gmag[i] = 1000.
    else:
        gmag[i] = tab_gmag_gmag[st1[0][0]]


f1 = open(os.getcwd() + '/Zilinskas/Analysis/gmag.dat', 'w')
for i in range(len(stars)):
    f1.write(stars[i] + '\t\t' + str(gmag[i]) + '\t\t' + str(subT[i]) + '\n')
f1.close()