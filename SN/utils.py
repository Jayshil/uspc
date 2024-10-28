import numpy as np
import astropy.constants as con
import astropy.units as u
from astropy.modeling.models import BlackBody

def planck_func(lam, temp):
    """
    Given the wavelength and temperature
    this function will compute the specific
    intensity using the Planck's law
    """
    coeff1 = (2 * con.h * con.c * con.c)/(lam**5)
    expo = np.exp( (con.h * con.c) / (lam * con.k_B * temp) ) - 1
    planck = (coeff1/expo).to(u.W / u.m**2 / u.micron)
    return planck

def planck_func_astropy(lam, temp):
    """
    Given the wavelength and temperature
    this function will compute the specific
    intensity using the Planck's law as defined 
    in astropy package -- will return the output 
    same as previous function
    """
    bb = BlackBody(temperature=6000*u.K)
    flux = bb(lam)
    flux = flux.to(u.W / u.m**2 / u.micron / u.sr, equivalencies=u.spectral_density(wav=lam))
    return flux