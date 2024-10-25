import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import matplotlib
import matplotlib.cm as cm
from utils import generate_occ_lc, lcbin
import juliet
import os

cheops_expected_dep = 39.696787188657325

# Defining colors from colormap (will define 10 colors -- and will choose 2 out of them)
chex = np.array([])
norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)
for i in range(11):
    c1 = cm.plasma(norm(100*i))     # Use cm.viridis for viridis colormap and so on...
    c2 = matplotlib.colors.rgb2hex(c1, keep_alpha=True)
    chex = np.hstack((chex, c2))

# Loading juliet data
dataset = juliet.load(input_folder=os.getcwd() + '/Simulations/Analysis/Realistic1807_40Obs/')
res = dataset.fit(sampler='dynamic_dynesty')

# Posteriors
post = res.posteriors['posterior_samples']
fp, tsec, per = post['fp_p1'], post['t_secondary_p1'], post['P_p1']
t0 = tsec - (per/2)

# Best-fitted model
dummy_tim = np.linspace(np.nanmedian(tsec)-(np.nanmedian(per)/4), np.nanmedian(tsec)+(np.nanmedian(per)/4), 1000)
dummy_phs = juliet.utils.get_phases(t=dummy_tim, P=0.549374, t0=2458899.3449, phmin=1.)
_, best_fit, _ = generate_occ_lc(tim=dummy_tim, per=0.549374, t0=2458899.3449, rprs=0.0182, bb=0.53, ar=3.8,\
                                 fp=np.nanmedian(fp), mflux=0., prec=0.)

# The figure
fig = plt.figure(figsize=(12,5.5))
gs = gd.GridSpec(1,2, width_ratios=[3,1], wspace=0.01)

# First the data
ax1 = plt.subplot(gs[0])
all_tim, all_fl, all_fle, all_phs = np.array([]), np.array([]), np.array([]), np.array([])
for i in range(40):
    # Loading the data
    tim1, fl1, fle1 = dataset.times_lc['TOI1807' + str(i)], dataset.data_lc['TOI1807' + str(i)], dataset.errors_lc['TOI1807' + str(i)]
    # Normalising the data
    fl1 = fl1 * (1 + np.nanmedian(post['mflux_TOI1807' + str(i)]))
    fl1 = (fl1 - 1.) * 1e6
    # Computing phases
    ph1 = juliet.utils.get_phases(tim1, P=0.549374, t0=2458899.3449, phmin=1.)
    ax1.errorbar(ph1, fl1, fmt='.', c='cornflowerblue', alpha=0.25, zorder=1)
    # Storing the values
    all_tim, all_fl, all_fle = np.hstack((all_tim, tim1)), np.hstack((all_fl, fl1)), np.hstack((all_fle, fle1))
    all_phs = np.hstack((all_phs, ph1))

# Binning
phbin, flbin, flebin, _ = lcbin(time=all_phs, flux=all_fl, binwidth=0.01)
ax1.errorbar(phbin, flbin, yerr=flebin, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
for i in range(50):
    _, random_fit, _ = generate_occ_lc(tim=dummy_tim, per=np.random.choice(per), t0=np.random.choice(t0), rprs=0.0182, bb=0.53, ar=3.8,\
                                       fp=np.random.choice(fp), mflux=0., prec=0.)
    random_fit = (random_fit - 1.)*1e6
    ax1.plot(dummy_phs, random_fit, color='orangered', zorder=75, lw=1, alpha=0.3)

# Models
ax1.plot(dummy_phs, (best_fit-1.)*1e6, color='navy', lw=2.5, zorder=100)

ax1.set_xlim([np.min(all_phs), np.max(all_phs)])
ax1.set_ylim([-75., 150.])
ax1.set_xlabel('Orbital phase', fontsize=16)
ax1.set_ylabel('Relative flux [ppm]', fontsize=16)
ax1.set_xticks(ticks=np.array([0.4, 0.425, 0.450, 0.475, 0.5, 0.525, 0.550, 0.575, 0.6]),\
               labels=np.array(['0.40', '', '0.45', '', '0.50', '', '0.55', '', '0.60']))

# And the posterior
ax2 = plt.subplot(gs[1])
ax2.hist(fp*1e6, bins=100, color='k', histtype='step', density=True);
ax2.axvline(cheops_expected_dep, lw=2., color=chex[2], label='Expected value')
ax2.legend(loc='best')
ax2.set_xlabel('Occultation depth [ppm]', fontsize=16)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_ylabel('Density', fontsize=16)


plt.setp(ax1.get_xticklabels(), fontsize=14)
plt.setp(ax2.get_xticklabels(), fontsize=14)
plt.setp(ax1.get_yticklabels(), fontsize=14)
plt.setp(ax2.get_yticklabels(), fontsize=14)

#plt.show()

plt.savefig(os.getcwd() + '/Simulations/Analysis/simulation.png', dpi=500)