import numpy as np
import matplotlib.pyplot as plt
import juliet
import os
from exotoolbox import plots
import matplotlib.gridspec as gd
import multiprocessing
multiprocessing.set_start_method('fork')

# Transit+PC only analysis

instruments = ['TESS14', 'TESS15', 'TESS16', 'TESS21', 'TESS22', 'TESS23', 'TESS41', 'TESS48', 'TESS49', 'TESS50', 'TESS75', 'TESS76', 'TESS77']
pout = os.getcwd() + '/TOI-1860/Analysis/WoPlanet'

# Loading the data
tim_all, fl_all, fle_all = {}, {}, {}
for i in range(len(instruments)):
    tim7, fl7, fle7 = np.loadtxt(os.getcwd() + '/Data/PDCSAP/TOI-1860/' + instruments[i] + '.dat', usecols=(0,1,2), unpack=True)
    tim_all[instruments[i]], fl_all[instruments[i]], fle_all[instruments[i]] = tim7, fl7, fle7


## Instrumental priors
par_ins, dist_ins, hyper_ins = [], [], []
for i in range(len(instruments)):
    par_ins = par_ins + ['mdilution_' + instruments[i], 'mflux_' + instruments[i], 'sigma_w_' + instruments[i]]
    dist_ins = dist_ins + ['fixed', 'normal', 'loguniform']
    hyper_ins = hyper_ins + [1.0, [0., 0.1], [0.1, 1e4]]

## GP priors
par_gp = ['GP_sigma_' + '_'.join(instruments), 'GP_timescale_' + '_'.join(instruments), 'GP_rho_' + '_'.join(instruments)]
dist_gp = ['loguniform', 'loguniform', 'loguniform']
hyper_gp = [[1e-5, 10000.], [1e-3, 1e2], [1e-3, 1e2]]

## Total priors
par_tot = par_ins + par_gp
dist_tot = dist_ins + dist_gp
hyper_tot = hyper_ins + hyper_gp

priors = juliet.utils.generate_priors(params=par_tot, dists=dist_tot, hyperps=hyper_tot)

# And, the juliet analysis
dataset = juliet.load(priors=priors, t_lc=tim_all, y_lc=fl_all, yerr_lc=fle_all, GP_regressors_lc=tim_all, out_folder=pout)
res = dataset.fit(sampler='dynamic_dynesty', nthreads=8)#, light_travel_delay=True, stellar_radius=0.690)


# Let's plot some cool results!
for i in range(len(instruments)):
    instrument = instruments[i]

    model = res.lc.evaluate(instrument)

    # Let's make sure that it works:
    fig = plt.figure(figsize=(16/1.5,9/1.5))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(tim_all[instrument], fl_all[instrument], yerr=fle_all[instrument], fmt='.', alpha=0.3)
    ax1.plot(tim_all[instrument], model, c='k', zorder=100)
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim_all[instrument]), np.max(tim_all[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(tim_all[instrument], (fl_all[instrument]-model)*1e6, yerr=fle_all[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (BJD)')
    ax2.set_xlim(np.min(tim_all[instrument]), np.max(tim_all[instrument]))
    #plt.show()
    plt.savefig(pout + '/full_model_' + instrument + '.png')

plots.corner_plot(pout)