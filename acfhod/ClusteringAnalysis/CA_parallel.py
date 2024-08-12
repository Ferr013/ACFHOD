###################################################################################################
# Giovanni Ferrami August 2024
###################################################################################################
import sys
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import uuid

import numpy as np
from scipy import special
from scipy.interpolate import splrep, splev, splint
import acfhod.Utils.Utils as utils
import acfhod.HOD.HOD as HOD
import acfhod.ClusteringAnalysis.CA as CA

cosmo, sigma_8 = utils.get_cosmology()
c_light  = 299792.458 #speed of light km/s

###################################################################################################
### CLUSTERING ANALYSIS PARALLEL CODE #############################################################

def init(mem):
    global mem_id
    mem_id = mem
    return

def calc_1halo_integral(t, j_0_zeros, k_array, comoving_distance_z,
                        PS_1_spl, STEP_J0, DELTA_J0, RES_J0, SPL_ORDER):
    res = 0
    k0 = j_0_zeros[0]/t/comoving_distance_z
    k_here = np.append(k_array[k_array<k0], k0)
    PS_1_here = splev(k_here, PS_1_spl)
    Bessel = np.array([special.j0(k*t*comoving_distance_z) for k in k_here])
    integrand = PS_1_here * k_here / (2*np.pi) * Bessel
    A_sp = splrep(k_here, integrand, s=0, k=SPL_ORDER)
    res += splint(k_here[0], k_here[-1], A_sp)
    i = 0
    while i <= STEP_J0 - DELTA_J0:
        j_array = np.linspace(j_0_zeros[i], j_0_zeros[i+DELTA_J0], DELTA_J0*RES_J0)
        k_here = j_array / t / comoving_distance_z
        PS_1_here = splev(k_here, PS_1_spl)
        Bessel = np.array([special.j0(k*t*comoving_distance_z) for k in k_here])
        integrand = PS_1_here * k_here / (2*np.pi) * Bessel
        A_sp = splrep(k_here, integrand, s=0, k=SPL_ORDER)
        res += splint(k_here[0], k_here[-1], A_sp)
        i += DELTA_J0
    return res

def omega_inner_integral(theta,
                        comoving_distance_z,
                        M_h_array, HMF_array,
                        NCEN, NSAT, U_FT, k_array, hmf_PS, bias,
                        STEP_J0 = 100_000, DELTA_J0 = 10_000, RES_J0 = 8):
    SPL_ORDER = 1
    PS_1   = HOD.PS_1h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    PS_2   = HOD.PS_2h(M_h_array, HMF_array, NCEN, NSAT, U_FT, bias)
    N1, N2 = np.max(PS_1), np.max(PS_2)
    PS_1 = PS_1 / N1
    PS_2 = PS_2 / N2
    j_0_zeros = special.jn_zeros(0, STEP_J0+1)
    PS_1_spl = splrep(k_array, PS_1, s=0, k=SPL_ORDER)
    R_T1 = np.array([calc_1halo_integral(t, j_0_zeros, k_array, comoving_distance_z,\
                    PS_1_spl, STEP_J0, DELTA_J0, RES_J0, SPL_ORDER) for t in theta])
    Bessel = np.array([\
             np.array([special.j0(k*t*comoving_distance_z) for k in k_array])\
             for t in theta])
    A_sp = [splrep(k_array, hmf_PS * PS_2 * k_array / (2*np.pi) * Bessel[it], s=0, k=1) \
            for it in range(len(theta))]
    R_T2 = np.array([splint(0, k_array[-1], A_sp[it]) for it in range(len(theta))])
    return R_T1 * N1, R_T2 * N2

def omega_z_component_single(args):
    job_id, shape, z, _args_ = args
    theta, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS, STEP_J0 = _args_
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    shres = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    M_h_array, HMF_array, nu_array, k_array, hmf_PS = \
        utils.init_lookup_table(z, M_DM_min, M_DM_max, REWRITE_TBLS)
    crit_dens_rescaled = (4/3*np.pi*cosmo.critical_density(z).value*200*2e40)
    U_FT = np.array([HOD.u_FT(k, M_h_array, z, crit_dens_rescaled) for k in k_array])
    bias = HOD.bias_Tinker10(nu_array)
    comoving_distance_z = cosmo.comoving_distance(z).value
    shres[job_id, 0, :], shres[job_id, 1, :] = omega_inner_integral(theta,
                                                comoving_distance_z,
                                                M_h_array, HMF_array,
                                                NCEN, NSAT, U_FT, k_array, hmf_PS, bias,
                                                STEP_J0 )
    return

def omega_z_component_parallel(z_array, theta_array, M_DM_min, M_DM_max, NCEN, NSAT,
                               REWRITE_TBLS = False, STEP_J0 = 50_000, cores=None):
    if cores is None:
        if len(z_array) <= multiprocessing.cpu_count():
            cores = len(z_array)
        else:
            print('MORE Z BINS THAN CORES, THE CODE IS NOT SMART ENOUGH TO HANDLE THIS YET')
            raise ValueError
            #cores = multiprocessing.cpu_count()
    _args_ = theta_array, M_DM_min, M_DM_max, NCEN, NSAT, REWRITE_TBLS, STEP_J0
    shape = (len(z_array), 2, len(theta_array))
    args =  [(i, shape, z_array[i], _args_) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        nbytes = (2 * len(z_array) * len(theta_array)) * np.float64(1).nbytes
        shd_mem = SharedMemory(name=f'{mem_id}', create=True, size=nbytes)
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(mem_id,))
        try:
            pool.map_async(omega_z_component_single, args, chunksize=1).get(timeout=10_000)
        except KeyboardInterrupt:
            print("Caught kbd interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            z_h_t_array = np.ndarray(shape, buffer=shd_mem.buf, dtype=np.float64).copy()
    finally:
        shd_mem.close()
        shd_mem.unlink()
        if exit:
            sys.exit(1)
    return z_h_t_array

def omega_1_and_2halo(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
                      mag_min = 0, mag_max = np.inf, STEP_J0 = 100_000,
                      REWRITE_TBLS = False, cores=None):
    if mag_min == 0 or mag_max == np.inf:
        M_DM_min, M_DM_max = 0, np.inf
    else:
        M_DM_min, M_DM_max = utils.get_M_DM_range(np.mean(z_array),
                                                  mag_max, mag_min,
                                                  delta_z=0.5)
    M_h_array, __, ___, ____, _____ =\
            utils.init_lookup_table(0, M_DM_min, M_DM_max, REWRITE_TBLS)
    NCEN = HOD.N_cen(M_h_array, M_min, sigma_logM)
    NSAT = HOD.N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    H_z = [cosmo.H(z).value for z in z_array]
    factor_z = np.power(np.array(N_z_nrm), 2) / (c_light / np.array(H_z))
    itg  = omega_z_component_parallel(z_array, theta,
                                      M_DM_min, M_DM_max,
                                      NCEN, NSAT,
                                      REWRITE_TBLS, STEP_J0, cores)
    N_G = HOD.get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
                             int_M_min=np.power(10, M_DM_min),
                             int_M_max=np.power(10, M_DM_max))
    I1 = np.array([np.trapz(itg[:,0,i] * factor_z, z_array) for i in range(len(theta))])
    I2 = np.array([np.trapz(itg[:,1,i] * factor_z, z_array) for i in range(len(theta))])
    return I1/ np.power(N_G, 2), I2/ np.power(N_G, 2)
