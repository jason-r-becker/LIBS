# LIBS
Analysis codes for Laser-Induced Breakdown Spectroscopy data

# Temperature
Utilizes Boltzmann method to estimate plasma excitation temperature
  temperature.py: calculates temperature using one spectra for a series of delays, useful for initial experiment
  Temperature Calculations.py: calculates average temperature for a series of delays with multiple trials, as well as different
                               energies/pulse types and compares them on one figure with error propogation

# Electron Density
Utilizes Stark broadening method with Voigt fitting to estimate plasma electron number density
  Voigt.py: uses lmfit library to calculate Lorentzian FWHM, associated error, and R^2 of fit for individual spectra
  edTest.py: calculates FWHM and electron density using one spectra for a series of delays, with ability to tune window size 
             used in Voigt fitting
  Electron Density.py: calculates average electron density for a series of delays with multiple trials, as well as different
                       energies/pulse types and compares them on one figure with error propogation
  fwhm_to_ED.py: calculates electron density from stark impact parameter (including temperature dependence if available) form FWHM values
  
# Plasma Persistence
Figures generated for multiple energies/pulse types using contour maps with signal-to-noise ratio as scale
