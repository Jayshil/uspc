import numpy as np
import matplotlib.pyplot as plt
import juliet
import os
from exotoolbox import plots
import astropy.units as u
import matplotlib.gridspec as gd
from utils import computeRMS
import multiprocessing
multiprocessing.set_start_method('fork')

# Transit with phase curve (phase offset != 0) analysis

instruments = ['TESS1', 'TESS27', 'TESS28', 'TESS68']
pout = os.getcwd() + '/HD213885/Analysis/PCwoPhaseOff'

# Loading the data
tim_all, fl_all, fle_all = {}, {}, {}
for i in range(len(instruments)):
    tim7, fl7, fle7 = np.loadtxt(os.getcwd() + '/Data/PDCSAP/HD213885/' + instruments[i] + '.dat', usecols=(0,1,2), unpack=True)
    tim_all[instruments[i]], fl_all[instruments[i]], fle_all[instruments[i]] = tim7, fl7, fle7

# Planetary parameters (From Espinoza et al. 2020)
per, per_err =  1.008035, np.sqrt((0.000021**2) + (0.000020**2))
tc, tc_err = 2458379.97043, 0.0012
ar, ar_err = 3.927, 0.037
bb, bb_err = 0.675, np.sqrt((0.033**2) + (0.041**2))
rprs, rprs_err = 0.01453, np.sqrt((0.00041**2) + (0.00042**2))

# Stellar parameters (It is derived from _stellar_ analysis by Espinoza et al. 2020; this is NOT transit density)
rho_st, rho_st_err = 1127, 33

## Computing transit time for TESS epoch
cycle = round((tim_all[instruments[-2]][0] - tc)/per)
tc1, tc1_err = tc + (cycle*per), tc_err + (cycle*per_err)
## Occultation time
tc2, tc2_err = tc1 + (0.5 * per), tc1_err + (0.5 * per_err)

# Priors
## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + '_'.join(instruments), 'q2_' + '_'.join(instruments), 'rho', 'ecc_p1', 'omega_p1']
dist_P = ['normal', 'normal', 'normal', 'normal', 'uniform', 'uniform', 'normal', 'fixed', 'fixed']
hyper_P = [[per, per_err], [tc1, tc1_err], [rprs, 5*rprs_err], [bb, 5*bb_err], [0., 1.], [0., 1.], [rho_st, rho_st_err], 0., 90.]

## Phase curve priors
par_pc = ['fp_p1', 't_secondary_p1', 'phaseoffset_p1']
dist_pc = ['uniform', 'normal', 'fixed']
hyper_pc = [[0.e-6, 500.e-6], [tc2, tc2_err], 0.]

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
par_tot = par_P + par_pc + par_ins + par_gp
dist_tot = dist_P + dist_pc + dist_ins + dist_gp
hyper_tot = hyper_P + hyper_pc + hyper_ins + hyper_gp

priors = juliet.utils.generate_priors(params=par_tot, dists=dist_tot, hyperps=hyper_tot)

# And, the juliet analysis
dataset = juliet.load(priors=priors, t_lc=tim_all, y_lc=fl_all, yerr_lc=fle_all, GP_regressors_lc=tim_all, out_folder=pout)
res = dataset.fit(sampler='dynesty', nthreads=8)


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

    # Allan deviation plot
    residuals = fl_all[instrument] - model
    rms, stderr, binsz = computeRMS(residuals, binstep=1)
    normfactor = 1e-6

    fig = plt.figure(figsize=(8,6))
    plt.plot(binsz, rms / normfactor, color='black', lw=1.5,
                    label='Fit RMS', zorder=3)
    plt.plot(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                    label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)
    plt.xlim(0.95, binsz[-1] * 2)
    plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
    plt.xlabel("Bin Size (N frames)", fontsize=14)
    plt.ylabel("RMS (ppm)", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(pout + '/alan_deviation_' + instrument + '.png')

plots.corner_plot(pout)