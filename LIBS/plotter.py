import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import prettyPrint as pp
import warnings

from matplotlib import style
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Settings
# ------------------------------------------------------------ #
sample = 'Si'
figure = 't'  # {'ed': electron density, 't': temperature, 'fwhm': FWHM}
save = True
drop_index = [15, 30]
# ------------------------------------------------------------ #

warnings.filterwarnings(action="ignore")
style.use('ggplot')

# Format filename from user inputs
if sample.lower() == 'mica':
    sample = 'Mica'
elif sample.lower() == 'ss':
    sample = 'SS'
elif sample.lower() == 'si':
    sample = 'Si'
else:
    print("Error: No data file exists for '{}'".format(sample))
    exit()

if figure.lower() == 'ed':
    figure = 'Electron Density'
    plt.xlabel('Delay Time (ns)')
    plt.ylabel('Electron Number Density ($1/cm^3$)')
    plt.xscale('log')
elif figure.lower() == 'fwhm':
    figure = 'FWHM'
    plt.xlabel('Delay Time (ns)')
    plt.ylabel('FWHM (nm)')
    plt.xscale('log')
elif figure.lower() in ['temp', 't']:
    figure = 'Temperature'
    plt.xlabel('Delay Time (ns)')
    plt.ylabel('Excitation Temperature (K)')
    plt.xscale('log')
elif figure.lower() in ['depth', 'd']:
    figure = 'Crater Depth'
    plt.xlabel('Number of Pulses')
    plt.ylabel('Crater Depth (μm)')
elif figure.lower() in ['volume', 'v']:
    figure = 'Crater Volume'
    plt.xlabel('Number of Pulses')
    plt.ylabel('Crater Volume ($μm^3$)')
else:
    print("Error: '{}' is not a valid entry".format(figure))
    exit()

# Read data
path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/Data/")
df = pd.read_excel('{}{} {}.xlsx'.format(path, sample, figure), index_col=0)
df = df.dropna(thresh=len(df))
if drop_index is not None:
    df = df.drop(drop_index)

pp.prettyPrint(df)

# Create Figure
pulse = ['gp', 'm1a', 'm1r', 'm2']
colors = ['black', 'orchid', 'firebrick', 'steelblue']
for p, c in zip(pulse, colors):
    plt.errorbar(df.index, df[p], yerr=df['{} err'.format(p)],
                 fmt='o', color=c, capsize=3, markeredgewidth=0.5, lw=0.5)
    xnew = np.linspace(df.index.min(), df.index.max(), 200)
    itp = interp1d(df.index, df[p], kind='linear')
    ysmooth = savgol_filter(itp(xnew), 101, 3)

    if len(df[p].dropna()) <= 3:
        mask = np.isfinite(np.array(df[p]).astype(np.double))
        plt.plot(df.index[mask], df[p][mask], color=c, lw=0.4)
    else:
        plt.plot(xnew, ysmooth, c=c, lw=0.4)

plt.title('{} {}'.format(sample, figure))
plt.legend(['Gaussian', 'M1, Azimuthal', 'M1, Radial', 'M2'])

# Save figure
if save:
    save_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/")
    plt.savefig('{}{} {}.eps'.format(save_path, sample, figure), bbox_inches='tight', format='eps', dpi=1000)
    print('Figure saved')

plt.show()
