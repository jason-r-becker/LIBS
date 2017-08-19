import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from lmfit.models import VoigtModel
from matplotlib import style
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

warnings.filterwarnings(action="ignore")
style.use('ggplot')

# Settings
delays = [15, 30, 50, 100, 200, 500]
span1 = (544.6, 545.0)
span2 = (545.4, 545.8)
span3 = (537.05, 537.5)

# Read data
def readData(delays):
    # Read full data
    full = pd.read_table('SS_ED_Test/15d 1000w_1.txt', header=None)

    # Read delay files
    df = pd.DataFrame()
    for d in delays:
        fid = pd.read_table('SS_ED_test/Delays_537/{}_1.txt'.format(d), header=None)
        if d == delays[0]:
            df['x'] = fid.iloc[:, 0]
        df['{}'.format(d)] = fid.iloc[:, 1]

    df.set_index(df['x'], inplace=True)
    df.drop('x', axis=1, inplace=True)

    return df, full


# Plot raw data
def plotRawData(df, full):
    plt.plot(full.iloc[:, 0], full.iloc[:, 1])
    plt.title('1000 ns width')
    plt.show()

    plt.plot(df.index.values, df['15'], 'red', df['30'], 'orange', df['50'], 'yellow', df['100'], 'green',
             df['200'], 'blue', df['500'], 'indigo')
    plt.legend(delays, title='Delay Time (ns)', loc='upper left')
    plt.show()


# Electron density calculation
def ed(FWHM, w):
    Hg = 0.0566
    return (FWHM - Hg) * 1e16 / 2 / w


def ed_err(error, w):
    return error * 1e16 / 2 / w


# Voigt fit routine
def voigtFit(df, delay, span, stats=False, plot=False):
    # Remove bad pixel
    df.drop(df.index[446], inplace=True)
    df.fillna(method='bfill', inplace=True)

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

    df = df[(df.index >= span[0]) & (df.index <= span[1])]
    x = df.index.values
    y = df[str(delay)].values

    # Set Voigt fit parameters
    mod = VoigtModel()
    pars = mod.guess(y, x=x)
    pars['gamma'].set(value=0.7, vary=True, expr='')
    # Perform Voigt fit
    out = mod.fit(y, pars, x=x)

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
        plt.plot(x, y, 'o', markersize=2.0, c='blue')
        plt.plot(x, out.best_fit, 'r-')
        try:
            dely = out.eval_uncertainty(sigma=5)
        except:
            dely = 0
        plt.fill_between(x, out.best_fit - dely, out.best_fit + dely, color="#bc8f8f")

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.xlim(span)

    # Save fit statistics
    for par_name, param in out.params.items():
        if par_name == 'gamma':
            return pd.DataFrame({'delay': [delay], 'fwhm_L': [2 * param.value],
                                 'error': [2 * param.stderr], 'R^2': [out.redchi]})


# Plot full results
def plotFitResults(plot_fits=False, stats=False):
    fe545 = pd.DataFrame()
    fe546 = pd.DataFrame()
    fe537 = pd.DataFrame()

    # Fill dataframes
    for d in delays:
        # fe545 = pd.concat([fe545, voigtFit(df, d, span1, plot=plot_fits)], ignore_index=True)
        # fe546 = pd.concat([fe546, voigtFit(df, d, span2, plot=plot_fits)], ignore_index=True)
        fe537 = pd.concat([fe537, voigtFit(df, d, span3, plot=plot_fits, stats=stats)], ignore_index=True)

        if plot_fits:
            plt.show()

    # Calculate electron density
    # fe545['ed'] = fe545['fwhm_L'].map(lambda x: ed(x, 0.0091))
    # fe546['ed'] = fe545['fwhm_L'].map(lambda x: ed(x, 0.0092))
    fe537['ed'] = fe537['fwhm_L'].map(lambda x: ed(x, 0.0049))
    fe537['ed_error'] = fe537['error'].map(lambda x: ed_err(x, 0.0049))

    # Plot results
    plt.subplot(1, 2, 1)
    # plt.plot(fe545['delay'], fe545['fwhm_L'], '-o', c='lightcoral')
    # plt.plot(fe546['delay'], fe546['fwhm_L'], '-o', c='mediumseagreen')
    plt.errorbar(fe537['delay'], fe537['fwhm_L'], yerr=fe537['error'], fmt='o', color='steelblue',
                 capsize=3, markeredgewidth=0.5, lw=0.5)
    xnew = np.linspace(np.array(delays).min(), np.array(fe537['delay']).max(), 1000)
    itp = interp1d(fe537['delay'], fe537['fwhm_L'], kind='linear')
    ysmooth = savgol_filter(itp(xnew), 101, 3)
    plt.plot(xnew, ysmooth, c='steelblue', lw=0.5)
    # plt.legend(['Fe I 544.69', 'Fe I 545.55'])
    plt.xlabel('Delay Time (ns)')
    plt.xscale('log')
    plt.ylabel('Lorentzian FWHM (nm)')
    plt.title('FWHM of Spectral Lines')

    plt.subplot(1, 2, 2)
    # plt.plot(fe545['delay'], fe545['ed'], '-o', c='lightcoral')
    # plt.plot(fe546['delay'], fe546['ed'], '-o', c='mediumseagreen')
    plt.errorbar(fe537['delay'], fe537['ed'], yerr=fe537['ed_error'], fmt='o', color='steelblue',
                 capsize=3, markeredgewidth=0.5, lw=0.5)
    xnew = np.linspace(np.array(delays).min(), np.array(fe537['delay']).max(), 1000)
    itp = interp1d(fe537['delay'], fe537['ed'], kind='linear')
    ysmooth = savgol_filter(itp(xnew), 101, 3)
    plt.plot(xnew, ysmooth, c='steelblue', lw=0.5)
    # plt.legend(['Fe I 544.69', 'Fe I 545.55'])
    plt.xlabel('Delay Time (ns)')
    plt.xscale('log')
    plt.ylabel('Electron Density ($1 / cm^3$)')
    plt.title('Electron Density of Plasma')

    plt.show()


# Plot raw data
df, full = readData(delays)
# plotRawData(df, full)
plotFitResults(plot_fits=False)
