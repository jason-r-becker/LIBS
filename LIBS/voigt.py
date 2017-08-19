import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import prettyPrint as pp

from lmfit.models import VoigtModel
from matplotlib import style

style.use('ggplot')
dataframe = pd.DataFrame()

pulsesList = ['gp', 'm1a', 'm1r', 'm2']
delaysList = range(1, 6)


# Perform Voigt fit of single spectra
def voigtFit(filename, xloc=0, yloc=1,  stats=False, plot=False):
    # Read Data
    df = pd.read_csv(filename, header=None)
    # Remove bad pixel
    df.drop(df.index[446], inplace=True)
    df.fillna(method='bfill', inplace=True)
    # Narrow region for later delays
    if 'd5' in filename:
        df = df[(df.iloc[:, xloc] > 287.75) & (df.iloc[:, xloc] < 288.6)]

    if 'd4' in filename and ('m1r' or 'm2' in filename):
        df = df[(df.iloc[:, xloc] > 287.75) & (df.iloc[:, xloc] < 288.6)]

    x = np.array(df.iloc[:, xloc])
    y = np.array(df.iloc[:, yloc])

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
        plt.plot(x, y, 'o', markersize=2.0, c='blue')
        plt.plot(x, out.best_fit, 'r-')
        dely = out.eval_uncertainty(sigma=5)
        plt.fill_between(x, out.best_fit - dely, out.best_fit + dely, color="#bc8f8f")
        plt.xlabel = 'Wavelength (nm)'
        plt.ylabel = 'Intensity (a.u.)'
        plt.xlim((287, 289.5))
        plt.show()

    # Save fit statistics
    for par_name, param in out.params.items():
        if par_name == 'gamma':
            return pd.DataFrame({'fid': [filename], 'fwhm_L': [2*param.value],
                                 'error': [2*param.stderr], 'R^2': [out.redchi]})


def pulses_delays_voigt(save_data=False, plot_data=False, data=dataframe, pulses=pulsesList, delays=delaysList):
    all_fits = data

    for pulse in pulses:
        for d in delays:
            for fid in [os.path.expanduser('~/odrive/LBL/Vortex Beam/6 21 2017 Si ED/4 Combination Location'
                                           ' Files/Si_{}_145uJ_d{}_ED.csv'.format(pulse, d))]:
                for y in range(1, 11):
                    all_fits = pd.concat([all_fits, voigtFit(fid, yloc=y)], ignore_index=True)

    dataMean = all_fits.groupby('fid').mean()
    dataSTD = all_fits.groupby('fid').std()
    final_data = dataMean

    final_data['error'] = dataSTD['error']
    cols = ['R^2', 'fwhm_L', 'error']
    final_data = final_data[cols]

    if save_data:
        final_data.to_csv('final_data.csv')

    if plot_data:
        # Remove bad points
        plot_df = final_data
        plot_df.loc[plot_df.fwhm_L < 0, 'fwhm_L'] = np.NaN
        plot_df.loc[plot_df.fwhm_L < 0, 'error'] = np.NaN
        plot_df.loc[plot_df.fwhm_L > 1, 'fwhm_L'] = np.NaN
        plot_df.loc[plot_df.fwhm_L > 1, 'error'] = np.NaN
        plot_df.loc[plot_df.error > 5, 'fwhm_L'] = np.NaN
        plot_df.loc[plot_df.error > 5, 'error'] = np.NaN

        gp = plot_df[plot_df.index.str.contains('_gp_')]
        m1r = plot_df[plot_df.index.str.contains('_m1r_')]
        m1a = plot_df[plot_df.index.str.contains('_m1a_')]
        m2 = plot_df[plot_df.index.str.contains('_m2_')]

        delay_vals = [15, 30, 50, 100, 200]
        plt.figure()
        plt.errorbar(delay_vals, gp['fwhm_L'], yerr=gp['error'], fmt='o', color='black',
                     capsize=3, markeredgewidth=0.5)
        plt.errorbar(delay_vals, m1r['fwhm_L'], yerr=m1r['error'], fmt='o', color='magenta',
                     capsize=3, markeredgewidth=0.5)
        plt.errorbar(delay_vals, m1a['fwhm_L'], yerr=m1a['error'], fmt='o', color='red',
                     capsize=3, markeredgewidth=0.5)
        plt.errorbar(delay_vals, m2['fwhm_L'], yerr=m2['error'], fmt='o', color='blue',
                     capsize=3, markeredgewidth=0.5)
        plt.xscale('log')
        plt.xlabel('Delay Time (ns)')
        plt.ylabel('FWHM (nm)')
        plt.title('BV-LIBS FWHM Values')
        plt.ylim([0, 1])
        plt.show()

    return all_fits, final_data


def delays_voigt(pulse='gp', save_data=False, plot_data=False, data=dataframe, delays=delaysList):
    all_fits = data

    for d in delays:
        fid = os.path.expanduser('~/odrive/LBL/Vortex Beam/6 21 2017 Si ED/4 Combination Location'
                                 ' Files/Si_{}_145uJ_d{}_ED.csv'.format(pulse, d))
        for y in range(1, 11):
            all_fits = pd.concat([all_fits, voigtFit(fid, yloc=y)], ignore_index=True)

    dataMean = all_fits.groupby('fid').mean()
    dataSTD = all_fits.groupby('fid').std()
    final_data = dataMean

    final_data['error'] = dataSTD['error']
    cols = ['R^2', 'fwhm_L', 'error']
    final_data = final_data[cols]

    if save_data:
        final_data.to_csv('final_data.csv')

    if plot_data:
        # Remove bad points
        plot_df = final_data
        plot_df.loc[plot_df.fwhm_L < 0, 'fwhm_L'] = np.NaN
        plot_df.loc[plot_df.fwhm_L < 0, 'error'] = np.NaN
        plot_df.loc[plot_df.fwhm_L > 1, 'fwhm_L'] = np.NaN
        plot_df.loc[plot_df.fwhm_L > 1, 'error'] = np.NaN
        plot_df.loc[plot_df.error > 5, 'fwhm_L'] = np.NaN
        plot_df.loc[plot_df.error > 5, 'error'] = np.NaN

        delay_vals = [15, 30, 50, 100, 200]
        plt.figure()
        plt.errorbar(delay_vals, plot_df['fwhm_L'], yerr=plot_df['error'], fmt='o', color='steelblue',
                     capsize=3, markeredgewidth=0.5)
        plt.xscale('log')
        plt.xlabel('Delay Time (ns)')
        plt.ylabel('FWHM (nm)')
        plt.title('{} FWHM Values'.format(pulse))
        plt.show()

    return all_fits, final_data


fid = os.path.expanduser('~/odrive/LBL/Vortex Beam/6 21 2017 Si ED/4 Combination Location'
                         ' Files/Si_gp_145uJ_d3_ED.csv')

# all_data, fit_data = pulses_delays_voigt(plot_data=True, save_data=True)
# all_data, fit_data = delays_voigt(pulse='m1a', plot_data=True)
fit_data = voigtFit(fid, stats=True, plot=True)

# pp.prettyPrint(all_data)
pp.prettyPrint(fit_data)

