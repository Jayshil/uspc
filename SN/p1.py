import numpy as np
from glob import glob
from astropy.table import Table
import astropy.units as u
import os

f1 = glob(os.getcwd() + '/Data/*_SN.votable')[0]
tab = Table.read(f1)

# Planet's name
name = np.asarray(tab['pl_name'])
name = np.array([name[i].replace(' ', '-') for i in range(len(name))])

# Stellar effective temperature
teff = np.asarray(tab['st_teff'])
# Planet's equilibrium temperature (from NEA)
teq_nea = np.asarray(tab['pl_eqt'])

# Other planetary parameters
per, gmag = np.asarray(tab['pl_orbper']), np.asarray(tab['sy_gaiamag'])
mp = np.asarray(tab['pl_bmasse'])

# Computing planetary equilibrium temperature
rprs = np.asarray(tab['pl_ratror'])
rp, rst = np.asarray(tab['pl_rade']), np.asarray(tab['st_rad'])

ar = np.asarray(tab['pl_ratdor'])
a = np.asarray(tab['pl_orbsmax'])

rprs_comps, ar_comps = np.copy(rprs), np.copy(ar)
for i in range(len(name)):
    if np.isnan(rprs[i]):
        if (not np.isnan(rp[i]))&(not np.isnan(rst[i])):
            rprs1 = ( (rp[i] * u.R_earth) / (rst[i] * u.R_sun) ).decompose()
            rprs_comps[i] = rprs1

for i in range(len(name)):
    if np.isnan(ar[i]):
        if (not np.isnan(a[i]))&(not np.isnan(rst[i])):
            ar1 = ( (a[i] * u.au) / (rst[i] * u.R_sun) ).decompose()
            ar_comps[i] = ar1

teq = np.zeros(len(name))
for i in range(len(name)):
    # Computing aRp
    r2a = 1 / (2 * ar_comps[i])
    if not np.isnan(r2a):
        teq[i] = teff[i] * np.sqrt(r2a)
    else:
        if not np.isnan(teq_nea[i]):
            teq[i] = teq_nea[i]
        else:
            teq[i] = np.nan

idx = ~np.isnan(teq)
idx[np.where(teq == 0.)] = False

# Removing all NaN values
name1, per1, rprs1, ar1 = name[idx], per[idx], rprs_comps[idx], ar_comps[idx]
rp1 = ((rprs_comps[idx] * rst[idx] * u.R_sun).to(u.R_earth)).value
a1 = ((ar_comps[idx] * rst[idx] * u.R_sun).to(u.au)).value
teq1, teff1 = teq[idx], teff[idx]
gmag1, mp1 = gmag[idx], mp[idx]

print(len(name1), len(per1), len(rprs1), len(ar1), len(rp1), len(a1), len(teq1), len(teff1), len(gmag1), len(mp1))

# Saving the data
f1 = open(os.getcwd() + '/SN/Analysis/planetary_data.dat', 'w')
f1.write('#Name\tPeriod\tRpRs\ta/Rs\tRpE\taAU\tTeqP\tTeffS\tGmag\tMpE\n')
for i in range(len(name1)):
    f1.write(str(name1[i]) + '\t' + str(per1[i]) + '\t' + str(rprs1[i]) + '\t' + str(ar1[i]) + '\t' + \
             str(rp1[i]) + '\t' + str(a1[i]) + '\t' + str(teq1[i]) + '\t' + str(teff1[i]) + '\t' + \
             str(gmag1[i]) + '\t' + str(mp1[i]) + '\n')
f1.close()