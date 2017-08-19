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
sample = 'SS'
fwhm_folder = r"8 1 2017 SS ED"
Hg = 0.05429
w_temp = np.asarray([0, 5, 10, 20, 40]) * 1e3
w_nm = np.asarray([0.0048])
save = True
# ------------------------------------------------------------ #

style.use('ggplot')
warnings.filterwarnings(action="ignore")

# Read data
temp = pd.read_excel(
    os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/Data/{} Temperature.xlsx".format(sample)),
    index_col=0)
fwhm = pd.read_excel(os.path.expanduser(r"~/odrive/LBL/Vortex Beam/{}/{} fwhm.xlsx".format(fwhm_folder, sample)),
                     index_col=0)

# Create ed dataframes
ed = pd.DataFrame(index=temp.index)

# Interpolate w
if len(w_nm) > 1:
    w = pd.DataFrame(index=temp.index)
    w_x = np.linspace(w_temp.min(), w_temp.max(), 101)
    itp = interp1d(w_temp, w_nm, kind='linear')
    w_y = savgol_filter(itp(w_x), 101, 3)

    # Plot w interpolation
    plt.plot(w_temp, w_nm, 'o', c='steelblue')
    plt.plot(w_x, w_y, lw=0.5, c='steelblue')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Stark Impact Parameter (nm)')
    plt.show()

    # Fill dataframes
    for p in [col for col in list(fwhm) if ('err' not in col) & (col != 'delay')]:  # just pulse columns; not errors
        w[p] = np.interp(temp[p], w_x, w_y)
        w[p + ' err'] = np.interp(temp[p] + temp[p + ' err'], w_x, w_y) - np.interp(temp[p], w_x, w_y)
        ed[p] = (fwhm[p] - Hg) * 1e16 / (2 * w[p])
        ed[p + ' err'] = ed[p] * np.sqrt((fwhm[p + ' err'] / fwhm[p]) ** 2 + (w[p + ' err'] / w[p]) ** 2)

else:
    w = w_nm[0]
    for p in [col for col in list(fwhm) if ('err' not in col) & (col != 'delay')]:  # just pulse columns; not errors
        ed[p] = (fwhm[p] - Hg) * 1e16 / (2 * w)
        ed[p + ' err'] = fwhm[p + ' err'] * 1e16 / (2 * w)


pp.prettyPrint(ed)

# Save results
if save:
    save_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/Data/")
    ed.to_excel('{}{} Electron Density.xlsx'.format(save_path, sample))
