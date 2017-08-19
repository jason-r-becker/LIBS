import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import prettyPrint as pp
import warnings

from matplotlib import style
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Settings
# ------------------------------------------------------------ #
folder = r"6 16 2017 Si Temp"
pulses = ['gp', 'm1a', 'm1r', 'm2']
delays = np.asarray([15, 30, 50, 100, 200, 500])  # (ns)
prec = 2  # peak center precision (nm)
individual_plots = False  # plot individual Boltzmann figures
save = True
save_fid = 'Si Temperature'
# ------------------------------------------------------------ #

warnings.filterwarnings(action="ignore")
style.use('ggplot')

# Fe line parameters [wavelength, A_ki, g_k, upper energy level (eV)]
fe = {
    'l5': [372.76, 0.225e8, 5, 4.28],
    'l7': [376.72, 0.64e8, 3, 4.30],
    'l11': [388.63, 5.29e6, 7, 3.24],
    'l13': [390.29, 2.14e7, 7, 4.73],
    'l14': [392.29, 1.08e6, 9, 3.21],
    'l15': [392.79, 2.60e6, 5, 3.26]
}
si = {'l1': [263.128, 1.06e8, 3, 6.619],
      'l2': [288.158, 2.17e8, 3, 5.082]}


# Read data files from source location
def read_data(pulse, d=delays, fldr=folder):
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
    for i in range(1, len(d) + 1):
        fid = r"{}Si_{}_145uJ_d{}_T.csv".format(path, pulse, i)
        temp = pd.read_csv(fid, header=None, index_col=0)
        df = df.join(temp, rsuffix='_{}'.format(str(i)))

    df.columns = range(df.shape[1])  # reset header
    df = df.div(correction.iloc[:, 0], axis=0)  # correct each spectra

    return df


# Boltzmann method for temperature calculation
def boltzmann(y, lines, spread=prec, plot=False):
    intensity = []
    e_k = []

    # Get Boltzmann plot values
    for line in lines.items():
        param = line[1]
        peak = max(y[(y.index > (param[0] - spread)) & (y.index < (param[0] + spread))])
        bkg = min(y[(y.index > (param[0] - 5)) & (y.index < (param[0] + 5))])
        b_intensity = np.log((peak - bkg) * param[0] / (param[1] * param[2]))
        intensity.append(b_intensity)
        e_k.append(param[3])

    # Fit Boltzmann line
    fit = np.polyfit(e_k, intensity, deg=1)
    fit_fn = np.poly1d(fit)
    m, b, r_val, p_val, std_err = stats.linregress(e_k, intensity)

    # Calculate temperature
    temp = -1 / (m * 8.617e-5)
    error = std_err / 8.617e-5

    if plot:
        plt.plot(e_k, intensity, 'o', c='steelblue')
        plt.plot(e_k, fit_fn(e_k), c='skyblue', lw=0.8)
        plt.xlabel('Upper Energy Level (eV)')
        plt.ylabel('ln($I*lambda/A*g$)')
        plt.title('Temperature: {:.0f} +/- {:.0f} K'.format(temp, error))
        plt.show()

    return temp, error


# Average temperature and error results for each accumulated spectra
def avg_results(df, lines, plot_boltz):
    temps = np.zeros(len(df.columns))
    errors = np.zeros(len(df.columns))
    for i in range(len(df.columns)):
        temps[i], errors[i] = boltzmann(df.iloc[:, i], lines, plot=plot_boltz)

    temps = np.reshape(temps, (-1, 10))
    errors = np.reshape(errors, (-1, 10))

    temps_mean = np.squeeze(np.asarray(np.mean(temps, axis=1)))
    temps_std = np.squeeze(np.asarray(np.std(temps, axis=1)))
    errors_sq_mean = np.squeeze(np.asarray(np.mean(errors ** 2, axis=1)))

    temps_err = (temps_std ** 2 + errors_sq_mean) ** 0.5

    return temps_mean, temps_std


colors = ['black', 'orchid', 'firebrick', 'steelblue']
tempDF = pd.DataFrame(index=delays)
for p, c in zip(pulses, colors):
    data = read_data(p)
    tempDF[p], tempDF['{} err'.format(p)] = avg_results(data, si, plot_boltz=individual_plots)
    plt.errorbar(delays, tempDF[p], yerr=tempDF['{} err'.format(p)],
                 fmt='o', color=c, capsize=3, markeredgewidth=0.5, lw=0.5)
    xnew = np.linspace(delays.min(), delays.max(), 500)
    itp = interp1d(delays, tempDF[p], kind='linear')
    ysmooth = savgol_filter(itp(xnew), 101, 3)
    plt.plot(xnew, ysmooth, c=c, lw=0.4)

pp.prettyPrint(tempDF)
plt.xlabel('Delay Time (ns)')
plt.xscale('log')
plt.legend(['GP', 'M1, Azimuthal', 'M1, Radial', 'M2'])
plt.ylabel('Temperature (K)')
plt.title(save_fid)

# Save data
if save:
    save_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/")
    tempDF.index.names = ['delay']
    tempDF.to_excel('{}{}.xlsx'.format(save_path, save_fid))
    plt.savefig('{}{}.eps'.format(save_path, save_fid), bbox_inches='tight', format='eps', dpi=1000)

plt.show()