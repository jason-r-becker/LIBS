import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
import prettyPrint as pp

from matplotlib import style


# Settings
# ------------------------------------------------------------ #
folders = [r"4 3 2017 Mica Temperature", r"6 16 2017 Si Temp", r"7 26 2017 SS Temp"]
sample_order = ['Mi', 'Si', 'SS']
ranges = [(300, 405), (255, 300), (370, 395)]
pulses = ['gp', 'm1a', 'm1r', 'm2']
delays = np.asarray([15, 30, 50, 100, 200, 500])  # (ns)
save = False
# ------------------------------------------------------------ #

warnings.filterwarnings(action="ignore")
style.use('ggplot')
colors = ['black', 'orchid', 'firebrick', 'steelblue']
samples = ['Mica', 'Si', 'Stainless Steel']


# Read data files from source location
def read_data(pulse, d, fldr, sample):
    path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/{}/4 Combination Location Files/".format(fldr))

    # Get x-axis wavelenth index
    df = pd.read_csv(path + os.listdir(path)[0], index_col=0, header=None)
    df = pd.DataFrame(index=df.index)

    # Read blackbody (bb) and collection optics (co) correction files
    bb_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Corrections/Deuterium Halogen.txt")
    bb = pd.read_table(bb_path, header=None, index_col=0)

    co_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/{}/H_D_1.txt".format(fldr))
    co = pd.read_table(co_path, header=None, index_col=0)

    co_in = np.interp(df.index, co.index, co.iloc[:, 0])  # interpolate bb curve to data's index
    bb_in = np.interp(df.index, bb.index, bb.iloc[:, 0])  # interpolate bb curve to data's index
    correction = np.asarray(co_in * bb_in)  # combine bb and co corrections
    correction = pd.DataFrame(correction, index=df.index)

    # Read data files
    for i in range(3, len(d) - 1):
        fid = r"{}{}_{}_145uJ_d{}_T.csv".format(path, sample, pulse, i)
        temp = pd.read_csv(fid, header=None, index_col=0)
        df = df.join(temp, rsuffix='_{}'.format(str(i)))

    df = df.div(correction.iloc[:, 0], axis=0)  # correct each spectra
    df = df.sum(axis=1)  # add all delays and locations together

    return df

# Plot all pulses for each sample
fig = plt.figure(figsize=(20, 9))
for j, s in zip(range(3), samples):
    plt.subplot(3, 1, j + 1)
    dfs = []
    max = np.zeros(4)
    for i in range(4):
        dfs.append(read_data(pulses[i], d=delays, sample=sample_order[j], fldr=folders[j]))
        dfs[i] = dfs[i][(dfs[i].index > ranges[j][0]) & (dfs[i].index < ranges[j][1])]
        max[i] = (dfs[i].max())

    norm = np.max(max)
    for i, c in zip(range(4), colors):
        dfs[i] /= norm
        plt.plot(dfs[i].index, dfs[i], color=c)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Signal Intensity (a.u.)')
    plt.title('  ')
    if j == 0:
        plt.text(307, 0.325, 'Al I 308.22')
        plt.text(307, 0.225, 'Al I 309.27')
        plt.text(389.5, 0.8, 'Al I 394.40')
        plt.text(396.4, 0.95, 'Al I 396.15')
        plt.text(ranges[j][0], 1.075, 'Mica', fontsize=16, horizontalalignment='left')
    elif j == 1:
        plt.text(261.8, 0.25, 'Si I 263.13')
        plt.text(285.3, 0.95, 'Si I 288.16')
        plt.text(ranges[j][0], 1.075, 'Si', fontsize=16, horizontalalignment='left')
    elif j == 2:
        plt.text(371.5, 0.65, 'Fe I 372.76')
        plt.text(375.2, 0.7, 'Fe I 376.72')
        plt.text(387.1, 0.8, 'Fe I 388.63')
        plt.text(389.6, 0.6, 'Fe I 390.29')
        plt.text(392.1, 0.425, 'Fe I 392.29')
        plt.text(392.8, 0.35, 'Fe I 392.79')
        plt.text(ranges[j][0], 1.075, 'Stainless Steel', fontsize=16, horizontalalignment='left')


plt.legend(['Gaussian', 'M1, Azimuthal', 'M1, Radial', 'M2'], loc='upper center', bbox_to_anchor=(0.5, -0.3),
           fancybox=True, shadow=True, ncol=4)
plt.tight_layout(pad=1, h_pad=1)

# Save figure
save_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/")
plt.savefig('{}Raw Data.eps'.format(save_path), bbox_inches='tight', format='eps', dpi=1000)
print('Figure saved')
plt.show()