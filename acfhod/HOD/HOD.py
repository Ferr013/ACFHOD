###################################################################################################
# Giovanni Ferrami July 2024
###################################################################################################
import numpy as np
from tqdm.notebook import tqdm
from scipy import special
from halomod.bias import Tinker10
from astropy.cosmology import Planck15
import acfhod.Utils.Utils as utils

cosmo = Planck15
sigma_8 = 0.8159
h = cosmo.H(0).value/100
c_light  = 299792.458 #speed of light km/s

def get_cosmology():
    return cosmo, sigma_8

###################################################################################################
### HALO OCCUPATION DISTRIBUTION ##################################################################
def N_cen(M_h, M_min, sigma_logM, DC = 1):
    return DC * 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    M_cut = np.power(M_min, -0.5) #Harikane 2018
    return DC * N_cen(M_h, M_min, sigma_logM) * np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM, DC = 1):
    return DC * (N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM))

def get_c_from_M_h(M_h, z, model='Correa'):
    #Correa ET AL: Eq.20 https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.1217C/abstract
    alpha = 1.226 - 0.1009*(1 + z) + 0.00378*(1 + z)**2
    beta  = 0.008634 - 0.08814*np.power((1 + z), -0.58816)
    log_c = alpha + beta * np.log10(M_h)
    return np.power(10, log_c)

def u_FT(k, M_h, z, crit_dens_rescaled):
    r_v = np.power(M_h/crit_dens_rescaled, 1/3) #rho = M_sun/Mpc^3
    c   = get_c_from_M_h(M_h, z)
    f_c = np.log(1+c)-c/(1+c)
    r_s = r_v/c
    si, ci = special.sici(k*r_s)
    si_c, ci_c = special.sici(k*r_s*(1+c))
    return (np.sin(k*r_s)*(si_c-si)+np.cos(k*r_s)*(ci_c-ci)-(np.sin(c*k*r_s)/((1+c)*k*r_s)))/f_c

def PS_1_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS1cs = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array) * 2
    PS1ss = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array) * 1
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array), 2)
    return np.array([(PS1cs + PS1ss), PS_2h])

def PS_1h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS1cs = np.trapz(HMF_array * NCEN * NSAT * U_FT, M_h_array) * 2
    PS1ss = np.trapz(HMF_array * NSAT * NSAT * U_FT * U_FT, M_h_array) * 1
    return PS1cs + PS1ss

def PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias):
    PS_2h = np.power(np.trapz((NCEN + NSAT) * HMF_array * bias * U_FT, M_h_array), 2)
    return PS_2h

###################################################################################################
### AVG QUANTITIES ################################################################################
def gal_density_n_g(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, DC = 1):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM, DC)
    return np.trapz(HMF_array*NTOT, M_h_array)

def get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                   DC = 1, int_M_min=0, int_M_max=np.inf):
    _N_G, _dVdz = np.zeros(0),  np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array, _, __, ___ = utils.init_lookup_table(z,
                                                                   M_DM_min = int_M_min,
                                                                   M_DM_max = int_M_max)
        m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
        _N_G  = np.append(_N_G, gal_density_n_g(M_min, sigma_logM, M_sat, alpha,
                                                M_h_array[m_mask], HMF_array[m_mask], DC))
        _dVdz = np.append(_dVdz, cosmo.comoving_distance(z).value**2 * c_light / cosmo.H(z).value)
    return np.trapz(_N_G * _dVdz * N_z_nrm, z_array)/np.trapz(_dVdz * N_z_nrm, z_array)

def get_AVG_N_tot(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array,
                  n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])/n_g
    return np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])/\
            np.trapz(HMF_array[m_mask], M_h_array[m_mask])

def get_AVG_Host_Halo_Mass(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array,
                           n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(M_h_array[m_mask]*HMF_array[m_mask]*NTOT, M_h_array[m_mask])/n_g
    return np.trapz(M_h_array[m_mask]*HMF_array[m_mask]*NTOT, M_h_array[m_mask])\
            /np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])

def get_EFF_gal_bias(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array, nu_array,
                     n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    bias = Tinker10(nu=nu_array[m_mask], sigma_8 = sigma_8, cosmo = cosmo).bias()
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(bias*HMF_array[m_mask]*NTOT, M_h_array[m_mask])/n_g
    return np.trapz(bias*HMF_array[m_mask]*NTOT, M_h_array[m_mask])/\
            np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])

def get_AVG_f_sat(M_min, sigma_logM, M_sat, alpha, M_h_array, HMF_array,
                  n_g=None, int_M_min=0, int_M_max=np.inf):
    m_mask = np.logical_and(M_h_array > int_M_min, M_h_array < int_M_max)
    NTOT = N_tot(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    NSAT = N_sat(M_h_array[m_mask], M_sat, alpha, M_min, sigma_logM)
    if n_g is not None:
        return np.trapz(HMF_array[m_mask]*NSAT, M_h_array[m_mask])/n_g
    return np.trapz(HMF_array[m_mask]*NSAT, M_h_array[m_mask])/\
            np.trapz(HMF_array[m_mask]*NTOT, M_h_array[m_mask])
###################################################################################################
