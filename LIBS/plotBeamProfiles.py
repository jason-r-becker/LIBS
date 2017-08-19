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
save = True
# ------------------------------------------------------------ #

warnings.filterwarnings(action="ignore")
style.use('ggplot')

# Read data
df = pd.read_excel(os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/Data/Beam Profiles.xlsx"))

# Create Figure
fig = plt.figure()
pulse = ['gp', 'm1a', 'm1r', 'm2']
colors = ['black', 'orchid', 'firebrick', 'steelblue']
for p, c in zip(pulse, colors):
    plt.plot(df['x ' + p] * 100, df[p], c=c)
plt.axhline(y=13.533, linewidth=0.8, ls='--', c='black')
ax = fig.add_subplot(111)
ax.annotate('$1/e^2$', xy=(115, 15))

plt.title('Beam Profiles')
plt.xlabel('Width (μm)')
plt.xlim(-140, 140)
plt.ylabel('Normalized Intensity')
plt.legend(['Gaussian', 'M1, Azimuthal', 'M1, Radial', 'M2'], loc='upper right')

# Save figure
if save:
    save_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/")
    plt.savefig('{}Beam Profiles.eps'.format(save_path), bbox_inches='tight', format='eps', dpi=1000)
    print('Figure saved')

plt.show()
