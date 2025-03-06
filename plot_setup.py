import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def set_plt_param(PLOT_FOR_KEYNOTE = 0):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['figure.figsize'  ] = (3.3,2.0)
    plt.rcParams['font.family'     ] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'       ] = 8
    plt.rcParams['axes.labelsize'  ] = 16
    plt.rcParams['legend.fontsize' ] = 13
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['legend.frameon'  ] = False
    plt.rcParams['xtick.labelsize' ] = 16
    plt.rcParams['ytick.labelsize' ] = 16
    plt.rcParams['xtick.direction' ] = 'in'
    plt.rcParams['ytick.direction' ] = 'in'
    plt.rcParams['xtick.top'       ] = True
    plt.rcParams['ytick.right'     ] = True
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1.25
    plt.rcParams['xtick.minor.width'] = 0.75
    plt.rcParams['ytick.major.width'] = 1.25
    plt.rcParams['ytick.minor.width'] = 0.75

    line_c = 'k'
    cmap_c = cm.inferno
    _col_  = None
    col_A  = 'k'
    col_B  = 'r'
    col_C  = 'm'
    col_D  = 'orange'
    fn_prefix = ''
    if PLOT_FOR_KEYNOTE:
        params_keynote = {
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "white",
        "axes.facecolor": '#222222',
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": 'white',
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": '#222222',
        "figure.edgecolor": 'lightgray',
        "savefig.facecolor": '#222222',
        "savefig.edgecolor": 'lightgray',
        }
        plt.rcParams.update(params_keynote)
        line_c = 'w'
        cmap_c = cm.cool
        _col_  = iter(cmap_c(np.linspace(0, 1, 4)))
        col_A  = next(_col_)
        col_B  = next(_col_)
        col_C  = next(_col_)
        col_D  = next(_col_)
        fn_prefix = 'KEY_'
    return line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix
