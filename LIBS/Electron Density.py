import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import prettyPrint as pp
import warnings

from lmfit.models import VoigtModel
from math import floor
from matplotlib import style
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Settings
# ------------------------------------------------------------ #
folder = "8 1 2017 SS ED"
pulses = ['gp', 'm1a', 'm1r', 'm2']
delays = [15, 30, 50, 100, 200, 500]  # (ns)
w = 0.09  # (nm) Stark impact parameter
individual_plots = False  # plot individual Voigt fits
save = True
save_fid = 'SS Electron Density'

# Wavelength spans for different delays
span1 = (544.6, 545.0)
span2 = (545.4, 545.8)
span3 = (537.05, 537.5)
# ------------------------------------------------------------ #

warnings.filterwarnings(action="ignore")
style.use('ggplot')

delays = np.asarray(delays)


# Read data files from source location
def read_data(pulse, d=delays, fldr=folder):
    path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/{}/4 Combination Location Files/".format(fldr))

    # Get x-axis wavelenth index
    df = pd.read_csv('{}{}'.format(path, os.listdir(path)[0]), index_col=0, header=None)
    df = pd.DataFrame(index=df.index)

    # Read data files
    for i in range(1, len(d) + 1):
        fid = r"{}SS_{}_145uJ_d{}_ED.csv".format(path, pulse, i)
        temp = pd.read_csv(fid, header=None, index_col=0)
        df = df.join(temp, rsuffix='_{}'.format(str(i)))

    df.columns = range(df.shape[1])  # reset header

    return df


# Voigt fitting to measure Lorentzian FWHM of spectral transition
def voigtFit(y, delay, span, w, stats=False, plot=False):
    # Remove bad pixel
    y.drop(y.index[446], inplace=True)

    # Limit fitting region to selected span
    if (delay == 15) & (span == (545.4, 545.8)):
        span = (545.55, 545.8)
    if (delay == 30) & (span == (545.4, 545.8)):
        span = (545.45, 545.85)
    elif (delay == 500) & (span == (545.4, 545.8)):
        span = (545.5, 545.75)

    if span == span3:
        if delay == 15:
            span = (536.9, 537.5)
        elif delay == 30:
            span = (537.05, 537.45)
        elif delay == 500:
            span = (537.1, 537.4)

    y_span = y[(y.index >= span[0]) & (y.index <= span[1])]

    x = y_span.index.values
    y_vals = y_span.values

    # Set Voigt fit parameters
    mod = VoigtModel()
    pars = mod.guess(y_vals, x=x)
    pars['gamma'].set(value=0.7, vary=True, expr='')
    # Perform Voigt fit
    out = mod.fit(y_vals, pars, x=x)

    # Print fit statistics
    if stats:
        print(out.fit_report(min_correl=0.25, show_correl=False))

    # Plot Voigt fit
    if plot:
        if span == span1:
            plt.subplot(1, 2, 1)
            plt.title('Fe I 544.69, Delay: {} ns'.format(delay))
        elif span == span2:
            plt.subplot(1, 2, 2)
            plt.title('Fe I 545.55, Delay: {} ns'.format(delay))
        else:
            plt.title('Fe I 537.15, Delay: {} ns'.format(delay))
        plt.plot(x, y_vals, 'o', markersize=2.0, c='blue')
        plt.plot(x, out.best_fit, 'r-')
        try:
            dely = out.eval_uncertainty(sigma=5)
        except:
            dely = 0
        plt.fill_between(x, out.best_fit - dely, out.best_fit + dely, color="#bc8f8f")

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.xlim(span)

    # Return FWHM values
    for par_name, param in out.params.items():
        if par_name == 'gamma':
            fwhm = 2 * param.value
            fwhm_err = 2 * param.stderr

    # Calculate electron density
    Hg = 0.0266
    ed = (fwhm - Hg) * 1e16 / (2 * w)
    ed_err = fwhm_err * 1e16 / (2 * w)

    return ed, ed_err


# Average temperature and error results for each accumulated spectra
def avg_results(df):
    ed = np.zeros(len(df.columns))
    errors = np.zeros(len(df.columns))
    for i in range(len(df.columns)):
        ed[i], errors[i] = voigtFit(df.iloc[:, i], delay=delays[floor(i / 10)], span=span1, w=w, plot=individual_plots)

    ed = np.reshape(ed, (-1, 10))
    errors = np.reshape(errors, (-1, 10))

    ed_mean = np.squeeze(np.asarray(np.mean(ed, axis=1)))
    ed_std = np.squeeze(np.asarray(np.std(ed, axis=1)))
    errors_sq_mean = np.squeeze(np.asarray(np.mean(errors ** 2, axis=1)))

    ed_err = (ed_std ** 2 + errors_sq_mean) ** 0.5

    return ed_mean, ed_std


colors = ['black', 'orchid', 'firebrick', 'steelblue']
ed_df = pd.DataFrame(index=delays)
for p, c in zip(pulses, colors):
    data = read_data(p)
    ed_df[p], ed_df['{} err'.format(p)] = avg_results(data)
    plt.errorbar(delays, ed_df[p], yerr=ed_df['{} err'.format(p)],
                 fmt='o', color=c, capsize=3, markeredgewidth=0.5, lw=0.5)
    xnew = np.linspace(np.array(delays).min(), np.array(delays).max(), 1000)
    itp = interp1d(delays, ed_df[p], kind='linear')
    ysmooth = savgol_filter(itp(xnew), 101, 3)
    plt.plot(xnew, ysmooth, c=c, lw=0.4)

pp.prettyPrint(ed_df)
plt.xlabel('Delay Time (ns)')
plt.xscale('log')
plt.legend(['GP', 'M1, Azimuthal', 'M1, Radial', 'M2'])
plt.ylabel('Electron Density ($1/cm^3$)')
plt.title(save_fid)

# Save data
if save:
    save_path = os.path.expanduser(r"~/odrive/LBL/Vortex Beam/Figure Data/")
    ed_df.index.names = ['delay']
    ed_df.to_excel('{}{}.xlsx'.format(save_path, save_fid))
    plt.savefig('{}{}.eps'.format(save_path, save_fid), bbox_inches='tight', format='eps', dpi=1000)

plt.show()
