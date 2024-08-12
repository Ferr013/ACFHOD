###################################################################################################
# Giovanni Ferrami August 2024
###################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import acfhod.Utils.Utils as utils
cosmo, sigma_8 = utils.get_cosmology()
c_light  = 299792.458 #speed of light km/s
###################################################################################################

def plot_ACF(theta, o1, o2, filename = None):
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3), sharex=False, sharey=False)
    ax.plot(theta*206265, o1, c = 'r', ls = '--', alpha = 0.5, label='1-halo')
    ax.plot(theta*206265, o2, c = 'b', ls = '--', alpha = 0.5, label='2-halo')
    ax.plot(theta*206265, o1+o2, c = 'k', ls = '-', alpha = 1, label='HOD')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((0.5,2e3))
    ax.set_ylim((1e-2, 1e1))
    ax.set_xlabel(r'$\theta$ [arcsec]')
    ax.set_ylabel(r'$\omega$($\theta$)')
    plt.legend()
    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    return
