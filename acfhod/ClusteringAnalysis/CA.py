###################################################################################################
# Giovanni Ferrami July 2024
###################################################################################################

import numpy as np
from scipy import special
from scipy.interpolate import splrep, splev, splint
from halomod.bias import Tinker10
import acfhod.HOD.HOD as HOD
import acfhod.Utils.Utils as utils

cosmo, sigma_8 = utils.get_cosmology()
c_light  = 299792.458 #speed of light km/s

###################################################################################################
### CLUSTERING ANALYSIS ###########################################################################

def omega_inner_integral_1halo(theta,
                               comoving_distance_z,
                               M_h_array, HMF_array,
                               NCEN, NSAT, U_FT, k_array, bias,
                               STEP_J0 = 100_000, DELTA_J0 = 10_000, RES_J0 = 8):
    SPL_ORDER = 1
    PS_1 = HOD.PS_1h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N1   = np.max(PS_1)
    PS_1 = PS_1 / N1
    PS_1_spl = splrep(k_array, PS_1  , s=0, k=SPL_ORDER)
    R_T1 = np.zeros(len(theta))
    j_0_zeros = special.jn_zeros(0, STEP_J0+1)
    for it, t in enumerate(theta):
        k0 = j_0_zeros[0]/t/comoving_distance_z
        k_here = np.append(k_array[k_array<k0], k0)
        PS_1_here = splev(k_here, PS_1_spl)
        Bessel = np.array([special.j0(k*t*comoving_distance_z) for k in k_here])
        integrand = PS_1_here * k_here / (2*np.pi) * Bessel
        A_sp = splrep(k_here, integrand, s=0, k=SPL_ORDER)
        R_T1[it] += splint(k_here[0], k_here[-1], A_sp)
        i = 0
        while i <= STEP_J0 - DELTA_J0:
            j_array = np.linspace(j_0_zeros[i], j_0_zeros[i+DELTA_J0], DELTA_J0*RES_J0)
            k_here = j_array / t / comoving_distance_z
            PS_1_here = splev(k_here, PS_1_spl)
            Bessel = np.array([special.j0(k*t*comoving_distance_z) for k in k_here])
            integrand = PS_1_here * k_here / (2*np.pi) * Bessel
            A_sp = splrep(k_here, integrand, s=0, k=SPL_ORDER)
            R_T1[it] += splint(k_here[0], k_here[-1], A_sp)
            i += DELTA_J0
    return R_T1 * N1

def omega_inner_integral_2halo(theta,
                               comoving_distance_z,
                               M_h_array, HMF_array,
                               NCEN, NSAT, U_FT, k_array, hmf_PS, bias):
    PS_2 = HOD.PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N2   = np.max(PS_2)
    PS_2 = PS_2 / N2
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    A_sp = [splrep(k_array, hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel[it], s=0, k=1) \
            for it in range(len(theta))]
    R_T2 = np.array([splint(0, k_array[-1], A_sp[it]) for it in range(len(theta))])
    return R_T2 * N2

def omega_z_component_1halo(z, args):
    theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS, STEP_J0 = args
    M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
        utils.init_lookup_table(z, M_DM_min, M_DM_max, REWRITE_TBLS)
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    U_FT = np.array([HOD.u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    return omega_inner_integral_1halo(theta, comoving_distance_z, M_h_array, HMF_array,
                                      NCEN, NSAT, U_FT, k_array, bias, STEP_J0)

def omega_z_component_2halo(z, args):
    theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS = args
    M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
        utils.init_lookup_table(z, M_DM_min, M_DM_max, REWRITE_TBLS)
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    U_FT = np.array([HOD.u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    return omega_inner_integral_2halo(theta, comoving_distance_z, M_h_array, HMF_array,
                                      NCEN, NSAT, U_FT, k_array, hmf_PS, bias)

def omega_z_component_1_and_2halo(z, args):
    theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS, STEP_J0 = args
    M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
        utils.init_lookup_table(z, M_DM_min, M_DM_max, REWRITE_TBLS)
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    U_FT = np.array([HOD.u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])
    bias = Tinker10(nu=nu_array, sigma_8 = sigma_8, cosmo = cosmo).bias()
    comoving_distance_z = cosmo.comoving_distance(z).value
    o1 = omega_inner_integral_1halo(theta, comoving_distance_z, M_h_array, HMF_array,
                                    NCEN, NSAT, U_FT, k_array, bias, STEP_J0)
    o2 = omega_inner_integral_2halo(theta, comoving_distance_z, M_h_array, HMF_array,
                                    NCEN, NSAT, U_FT, k_array, hmf_PS, bias)
    return o1, o2

def omega_1halo(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                mag_min = 0, mag_max = np.inf, STEP_J0 = 100_000, REWRITE_TBLS = False):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = utils.get_M_DM_range(np.mean(z_array), mag_max, mag_min, delta_z=0.5)
    M_h_array, __, ___, ____, _____ =\
            utils.init_lookup_table(0, M_DM_min, M_DM_max, REWRITE_TBLS)
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    args = theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS, STEP_J0
    itg = np.array([omega_z_component_1halo(z, args) for z in z_array])
    N_G = HOD.get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                             int_M_min=np.power(10, M_DM_min),
                             int_M_max=np.power(10, M_DM_max))
    I1 = np.array([np.trapz(itg[:,i] * factor_z, z_array) for i in range(len(theta))])
    return I1/ np.power(N_G, 2)

def omega_2halo(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                mag_min = 0, mag_max = np.inf, REWRITE_TBLS = False):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = utils.get_M_DM_range(np.mean(z_array), mag_max, mag_min, delta_z=0.5)
    M_h_array, __, ___, ____, _____ =\
            utils.init_lookup_table(0, M_DM_min, M_DM_max, REWRITE_TBLS)
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    args = theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS
    itg = np.array([omega_z_component_2halo(z, args) for z in z_array])
    N_G = HOD.get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                             int_M_min=np.power(10, M_DM_min),
                             int_M_max=np.power(10, M_DM_max))
    I2 = np.array([np.trapz(itg[:,i] * factor_z, z_array) for i in range(len(theta))])
    return I2/ np.power(N_G, 2)

def omega_1_and_2halo(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                      mag_min = 0, mag_max = np.inf, STEP_J0 = 100_000, REWRITE_TBLS = False):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = utils.get_M_DM_range(np.mean(z_array), mag_max, mag_min, delta_z=0.5)
    M_h_array, __, ___, ____, _____ =\
            utils.init_lookup_table(0, M_DM_min, M_DM_max, REWRITE_TBLS)
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    args = theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS, STEP_J0
    itg  = np.array([omega_z_component_1_and_2halo(z, args) for z in z_array])
    itg1, itg2 = itg[:,0], itg[:,1] #TODO: Check this
    N_G = HOD.get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                             int_M_min=np.power(10, M_DM_min),
                             int_M_max=np.power(10, M_DM_max))
    I1 = np.array([np.trapz(itg1[:,i] * factor_z, z_array) for i in range(len(theta))])
    I2 = np.array([np.trapz(itg2[:,i] * factor_z, z_array) for i in range(len(theta))])
    return I1/ np.power(N_G, 2), I2/ np.power(N_G, 2)
