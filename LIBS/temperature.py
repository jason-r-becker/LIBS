import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from matplotlib import style
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
style.use('ggplot')

# Settings
delays = [15, 30, 50, 100, 200, 500, 1000]

# Read data
df = pd.DataFrame()
for d in delays:
    fid = pd.read_table('temp_data/{}_1.txt'.format(d), header=None)
    if d == delays[0]:
        df['x'] = fid.iloc[:, 0]
    df['{}'.format(d)] = fid.iloc[:, 1]

df.set_index(df['x'], inplace=True)
df.drop('x', axis=1, inplace=True)
x = df.index.values


# Plot raw data
def plotRawData(x=x, df=df):
    plt.plot(x, df['15'], 'red', df['30'], 'orange', df['50'], 'yellow', df['100'], 'green',
             df['200'], 'blue', df['500'], 'indigo', df['1000'], 'violet')
    plt.legend(delays, title='Delay Time (ns)', loc='upper left')
    plt.show()


# Fe line parameters [wavelength, A_ki, g_k, upper energy level (eV)]
fe = {'l1': [374.949, 7.633e7, 9, 4.221],
      'l2': [375.823, 6.336e7, 7, 4.257],
      'l3': [381.584, 1.305e8, 7, 4.733],
      'l4': [382.043, 6.674e7, 9, 4.104],
      'l5': [382.588, 5.975e7, 7, 4.155]}


# Boltzmann method for temperature calculation
def boltzmann(lines, y, delay, spread=0.1, plot=False):
    intensity = []
    e_k = []

    # Get Boltzmann plot values
    for line in lines.items():
        param = line[1]
        peak = max(y[(y.index > (param[0] - spread)) & (y.index < (param[0] + spread))])
        b_intensity = np.log(peak * param[0] / (param[1] * param[2]))
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
        plt.plot(e_k, intensity, 'go', e_k, fit_fn(e_k), 'red', lw=0.3)
        plt.xlabel('Upper Energy Level (eV)')
        plt.ylabel('ln(I*lambda / A*g )')
        plt.title('Delay Time: {} ns\nTemperature: {:.0f} +/- {:.0f} K'.format(delay, temp, error))
        plt.show()

    return temp, error


plotRawData()

# Run for each delay
temps = []
errors = []
for i, d in enumerate(delays):
    temp, error = boltzmann(fe, y=df.iloc[:, i], delay=delays[i], plot=True)
    temps.append(temp)
    errors.append(error)

# Plot Results
plt.errorbar(delays, temps, yerr=errors, fmt='o', color='steelblue', capsize=3, markeredgewidth=0.5, lw=0.5)
plt.ylabel('Excitation Temperature (K)')
plt.xlabel('Delay Time (ns)')
plt.xscale('log')
xnew = np.linspace(np.array(delays).min(), np.array(delays).max(), 1000)
itp = interp1d(delays, temps, kind='linear')
ysmooth = savgol_filter(itp(xnew), 101, 3)
plt.plot(xnew, ysmooth, ls='dashed', c='lightcoral')
plt.show()
