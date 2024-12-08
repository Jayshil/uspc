import numpy as np
import batman

def generate_occ_lc(tim, per, t0, rprs, bb, ar, fp, mflux, prec):
    # First generating a deterministic light curve
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rprs
    params.a = ar
    params.inc = np.rad2deg(np.arccos(bb/ar))
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"
    params.fp = fp
    params.t_secondary = t0 + (per/2)
    m = batman.TransitModel(params, tim, transittype="secondary")
    flux = m.light_curve(params)
    deterministic = flux / (1 + mflux)

    # And now simulated
    fl, fle = np.zeros(len(tim)), np.zeros(len(tim))
    for i in range(len(tim)):
        fl[i] = np.random.normal(deterministic[i], prec)
        fle[i] = np.random.normal(prec, 0.1*prec)
    return tim, fl, fle

def generate_occ_lc_with_gaps(times, per, t0, rprs, bb, ar, fp, mflux, prec, eff):
    # Generating gappy times
    ## Computing total number of orbits in the data
    cheops_orbit_time_day = 98.77 / (60 * 24)
    orbit_nos = np.ptp(times) / cheops_orbit_time_day

    ## Generating roll numbers
    roll = np.linspace(0, orbit_nos*360, len(times)) + np.random.randint(0,360)
    roll = roll % 360
    idx_rollsort = np.argsort(roll)
    roll_rollsort = roll[idx_rollsort]
    tim_rollsort = times[idx_rollsort]

    ## Selecting the range of roll-angles to discard
    st_roll = np.random.choice(roll_rollsort)    # Starting roll number of discarded roll angles
    loc_st_roll = np.where(roll_rollsort == st_roll)[0][0]
    nos_discarded_pts = int((1 - (0.01 * eff)) * len(times)) # Total number of discarded points
    idx_discarded = np.ones(len(times), dtype=bool)

    if int(loc_st_roll+nos_discarded_pts) > len(times):
        # Roll over the starting roll numbers
        idx_discarded[loc_st_roll:] = False
        dis_pt = len(times) - loc_st_roll
        idx_discarded[0:nos_discarded_pts-dis_pt] = False
    else:
        idx_discarded[loc_st_roll:loc_st_roll + nos_discarded_pts] = False

    # Let's actually discard the points now
    tim = tim_rollsort[idx_discarded]

    # And sort them
    idx_timsort = np.argsort(tim)
    tim = tim[idx_timsort]

    # First generating a deterministic light curve
    params = batman.TransitParams()
    params.t0 = t0
    params.per = per
    params.rp = rprs
    params.a = ar
    params.inc = np.rad2deg(np.arccos(bb/ar))
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1, 0.3]
    params.limb_dark = "quadratic"
    params.fp = fp
    params.t_secondary = t0 + (per/2)
    m = batman.TransitModel(params, tim, transittype="secondary")
    flux = m.light_curve(params)
    deterministic = flux / (1 + mflux)

    # And now simulated
    fl, fle = np.zeros(len(tim)), np.zeros(len(tim))
    for i in range(len(tim)):
        fl[i] = np.random.normal(deterministic[i], prec)
        fle[i] = np.random.normal(prec, 0.1*prec)
    return tim, fl, fle

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    This code is taken from the code `pycheops`
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]