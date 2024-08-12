###################################################################################################
# Giovanni Ferrami July 2024
###################################################################################################
import os
import numpy as np
from hmf import MassFunction
import gzip
from astropy.cosmology import Planck15

BASEPATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/')

cosmo = Planck15
sigma_8 = 0.8159
h = cosmo.H(0).value/100
def get_cosmology():
    return cosmo, sigma_8
###################################################################################################
### Luminosity -> Halo Mass from Tacchella, Trenti et al. 2015 ####################################
def load_Tacchella_table():
    file_path = BASEPATH + "Tacchella_Trenti_Carollo_15.dat.gz"
    z, p, mag_d, mag, Mstar, Mdm, N = [], [], [], [], [], [], []
    with gzip.open(file_path, 'rt') as file:
        for line in file:
            columns = line.strip().split(' ')
            z.append(float(columns[0]))
            p.append(float(columns[1]))
            mag_d.append(float(columns[2]))
            mag.append(float(columns[3]))
            Mstar.append(float(columns[4]))
            Mdm.append(float(columns[5]))
            N.append(float(columns[7]))
    z, p, mag_d = np.array(z), np.array(p), np.array(mag_d)
    mag, Mstar, Mdm = np.array(mag), np.array(Mstar), np.array(Mdm)
    N = np.array(N)
    return z, p, mag_d, mag, Mstar, Mdm, N

def get_M_DM_range(z_analysis=5, m_max=-15, m_min=-22, delta_z=0.5):
    z, p, mag_d, mag, Mstar, Mdm, N = load_Tacchella_table()
    zmax, zmin = z_analysis + delta_z, z_analysis - delta_z
    if m_max < 0: #check if abs magnitudes
        _m_max,_m_min = np.max((m_max, m_min)), np.min((m_max, m_min))
        mag_max, mag_min = _m_max, _m_min
    else:
        _m_max,_m_min = np.max((m_max, m_min)), np.min((m_max, m_min))
        _distmd = 2.5 * np.log10(1+z_analysis) - cosmo.distmod(z_analysis).value
        mag_max, mag_min = _m_max + _distmd, _m_min + _distmd
    idx = np.where((z>=zmin) & (z<zmax) & (p==max(p)) & (mag<mag_max) & (mag>mag_min))[0]
    if len(idx) < 2:
        # print('The redshift and/or mass interval requested are not in the lookup table')
        if z_analysis > 1:
            # print('Trying z-0.5 --> z : ', z_analysis - 0.5)
            return get_M_DM_range(z_analysis - 1, m_max, m_min, delta_z)
        return -99, -99
    magg, mmdm = mag[idx], Mdm[idx]
    idx_sort   = np.argsort(magg)
    magg, mmdm = magg[idx_sort], mmdm[idx_sort]
    return np.log10(min(mmdm)), np.log10(max(mmdm))

###################################################################################################
#### INITIALIZE HMF ###############################################################################
def init_lookup_table(z, M_DM_min = 0, M_DM_max = np.inf, REWRITE_TBLS = False):
    FOLDERPATH = BASEPATH + 'HMF_Tables'
    min_lnk, max_ln_k, step_lnk = -11.5, 16.6, 0.05
    if os.path.exists(FOLDERPATH):
        FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'.txt'
        if (os.path.isfile(FPATH) and not REWRITE_TBLS):
            hmf_mass, hmf_dndm, hmf_nu = np.loadtxt(FPATH, delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_PS.txt'
            hmf_k, hmf_PS = np.loadtxt(FPATH, delimiter=',')
        else:
            print(f'Calculating HMF table at redshift {z:.2f}')
            hmf = MassFunction(Mmin = 9, Mmax = 17, dlog10m = 0.025,
                               lnk_min = min_lnk, lnk_max = max_ln_k, dlnk=step_lnk,
                               z=z, hmf_model = "Behroozi", sigma_8 = sigma_8, cosmo_model = cosmo)
            hmf_mass = hmf.m / h
            hmf_dndm = hmf.dndm * h**4
            hmf_nu   = hmf.nu
            np.savetxt(FPATH, (hmf_mass, hmf_dndm, hmf_nu),  delimiter=',')
            rd_st = 'redshift_' + str(int(z)) + '_'+str(int(np.around(z%1, 2) * 100)) + '_PS.txt'
            FPATH = FOLDERPATH + rd_st
            hmf_k = hmf.k * h
            hmf_PS_Nln = hmf.nonlinear_power / h**3
            hmf_PS = hmf_PS_Nln
            np.savetxt(FPATH, (hmf_k, hmf_PS),  delimiter=',')
            FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1,2)*100))+'_U.txt'
        m_mask = np.logical_and(np.log10(hmf_mass) > M_DM_min, np.log10(hmf_mass) < M_DM_max)
        return hmf_mass[m_mask], hmf_dndm[m_mask], hmf_nu[m_mask], hmf_k, hmf_PS
    else:
        print(FOLDERPATH)
        raise ValueError('Folder does not exist.')
###################################################################################################
