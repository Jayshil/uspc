import numpy as np
import matplotlib.pyplot as plt
from exotoolbox.utils import tdur
from exotoolbox import plots
import matplotlib.gridspec as gd
import juliet
import utils
import os

# Simulations are for TOI-1807 b

# Precision that we can get on 1 min data point is: 302.73 ppm (ETC). We will round it to 350 ppm (to account for gaps)
# I will simulate 30, 35 and 40 observations with 1 min cadence and preccision of 350 ppm.
# And I will perform a joint analysis of all that data

pout = os.getcwd() + '/Simulations/Analysis/Realistic1807_gaps_40Obs'
nobs = 30
fpfs = 39.7*1e-6
prec = 302.73*1e-6
visit_length = 2.6
eff = 60

# Planetary parameters (From Nardiello et al. 2022)
per, per_err =  0.549374, np.sqrt((0.000010**2) + (0.000013**2))
tc, tc_err = 2458899.3449, np.sqrt((0.0008**2) + (0.0005**2))
ar, ar_err = 3.8, 0.2
bb, bb_err = 0.53, np.sqrt((0.09**2) + (0.11**2))
rprs, rprs_err = 0.0182, 0.0006

tsec, tsec_err = tc + (per/2), np.sqrt(tc_err**2 + (per_err/2)**2)

t14 = tdur(per=per, ar=ar, rprs=rprs, bb=bb)

# ------------------------------------------------------------------------------
#
#                              Simulation
#
# ------------------------------------------------------------------------------

# Simulated times
cadence = 1 / (60 * 24)
cheops_orbit_time_day = 98.77 / (60 * 24)

times = np.arange(tsec-cheops_orbit_time_day*visit_length/2, tsec+cheops_orbit_time_day*visit_length/2, cadence)

# Saving data in a dictionary so that juliet understands
tim, fl, fle = {}, {}, {}


for obs in range(nobs):
    tim1, fl1, fle1 = utils.generate_occ_lc_with_gaps(times=times+(obs*per), per=per, t0=tc, rprs=rprs, bb=bb, ar=ar, fp=fpfs, mflux=np.random.normal(0., 0.0001), prec=prec, eff=eff)
    tim['TOI1807' + str(obs)], fl['TOI1807' + str(obs)], fle['TOI1807' + str(obs)] = tim1, fl1, fle1

# ------------------------------------------------------------------------------
#
#                              Retrieval
#
# ------------------------------------------------------------------------------

# Priors
## Planetary priors
par_P = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'a_p1', 'ecc_p1', 'omega_p1', 't_secondary_p1', 'fp_p1']
dist_P = ['normal', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'fixed', 'normal', 'uniform']
hyper_P = [[per, per_err], tc, rprs, bb, ar, 0., 90., [tsec, tsec_err], [0.e-6, 100.e-6]]

par_ins, dist_ins, hyper_ins = [], [], []
for i in range(nobs):
    par_ins = par_ins + ['mdilution_' + 'TOI1807' + str(i), 'mflux_' + 'TOI1807' + str(i), 'sigma_w_' + 'TOI1807' + str(i)]
    dist_ins = dist_ins + ['fixed', 'normal', 'loguniform']
    hyper_ins = hyper_ins + [1.0, [0., 0.1], [0.1, 10000.]]

priors = juliet.utils.generate_priors(par_P+par_ins, dist_P+dist_ins, hyper_P+hyper_ins)

# And, fitting
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, out_folder=pout)
res = dataset.fit(sampler='dynamic_dynesty', nthreads=8)

# And some plotting
for i in range(nobs):
    instrument = 'TOI1807' + str(i)

    model = res.lc.evaluate(instrument)

    # Let's make sure that it works:
    fig = plt.figure(figsize=(16/1.5,9/1.5))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.')#, alpha=0.3)
    ax1.plot(tim[instrument], model, c='k', zorder=100)
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.')#, alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (BJD)')
    ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    #plt.show()
    plt.savefig(pout + '/full_model_' + instrument + '.png')

plots.corner_plot(pout, True)