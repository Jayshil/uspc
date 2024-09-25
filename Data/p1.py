import numpy as np
from astroquery.mast import Observations
import os

# Path of the current directory
p1 = os.getcwd()

# Function to download TPF files
def tess_tpf_data(name, verbose=True):
    try:
        obt = Observations.query_object(name, radius=0.001)
    except:
        raise Exception('The name of the object does not seem to be correct.\nPlease try again...')
    # b contains indices of the timeseries observations from TESS
    b = np.array([])
    for j in range(len(obt['intentType'])):
        if obt['obs_collection'][j] == 'TESS' and obt['dataproduct_type'][j] == 'timeseries':
            b = np.hstack((b,j))
    if len(b) == 0:
        raise Exception('No TESS timeseries data available for this target.\nTry another target...')
    # To extract obs-id from the observation table
    sectors, pi_name, obsids, exptime, new_b = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(len(b)):
        data1 = obt['dataURL'][int(b[i])]
        if data1[-9:] == 's_lc.fits':
            fls = data1.split('-')
            for j in range(len(fls)):
                if len(fls[j]) == 5:
                    sec = fls[j]
            sectors = np.hstack((sectors, sec))
            new_b = np.hstack((new_b, b[i]))
            obsids = np.hstack((obsids, obt['obsid'][int(b[i])]))
            pi_name = np.hstack((pi_name, obt['proposal_pi'][int(b[i])]))
            exptime = np.hstack((exptime, obt['t_exptime'][int(b[i])]))
    if verbose:
        print('Data products found over sector(s): ', [str(i[-2:]) for i in sectors])
        print('Downloading them...')

    # Moving files to desired directory
    try:
        os.system('mkdir ' + p1 + '/Data/TPFs/' + name.replace(' ', '') + '/')
    except:
        pass
    
    for i in range(len(sectors)):
        dpr = Observations.get_product_list(obt[int(new_b[i])])
        cij = 0
        for j in range(len(dpr['obsID'])):
            if dpr['description'][j] == 'Target pixel files':
                cij = j
        tab = Observations.download_products(dpr[cij])
        lpt = tab['Local Path'][0][1:]
        
        os.system('mv ' + p1 + lpt + ' ' + p1 + '/Data/TPFs/' + name.replace(' ', '') + '/')
        os.system('rm -rf ' + p1 + lpt)

tess_tpf_data('HD 20329')
tess_tpf_data('HD 213885')