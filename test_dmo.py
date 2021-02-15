import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import seaborn as sns

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo


V = 100**3 # Mpc^3
output = 'output/'


## Load DMO simulation

# dmo = pd.read_csv('output/PMillennium_z000p000_dmo.csv')
dmo_pmill = pd.read_csv('output/PMillennium_z000p000_dmo_subset.csv')
pmill_V = (100 / 0.6777)**3 # (542.16 / 0.6777)**3
# dmo = dmo[dmo['FOF_Group_M_Crit200_DM'] > 2e9].reset_index()

dmo_L0100 = pd.read_csv('output/L0100N1504_028_z000p000_dmo.csv')
dmo_L0050 = pd.read_csv('output/L0050N0752_028_z000p000_dmo.csv')


def calc_df(x, binLimits, volume):
    hist, dummy = np.histogram(x, bins = binLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (binLimits[1] - binLimits[0])
    phi_sigma = (np.sqrt(hist) / volume) /\
                (binLimits[1] - binLimits[0]) # Poisson errors
    return phi, phi_sigma


## Subhalo mass function
binLimits = np.linspace(7.9, 16.9, 31)
bins = np.linspace(8.05, 16.75, 30)
lw = 3

key = 'M_DM'
phi, phi_sigma = calc_df(np.log10(dmo_pmill[key]), binLimits, pmill_V)
plt.plot(bins, np.log10(phi), label='P-Millennium', lw=lw)

phi, phi_sigma = calc_df(np.log10(dmo_L0100[key]), binLimits, 100**3)
plt.plot(bins, np.log10(phi), label='L0100N1504', lw=lw)

phi, phi_sigma = calc_df(np.log10(dmo_L0050[key]), binLimits, 50**3)
plt.plot(bins, np.log10(phi), label='L0050N0752', lw=lw)

plt.legend()
plt.xlim(8,)
plt.xlabel('$\mathrm{log_{10}}(M_{\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}})$')
plt.ylabel('$\mathrm{log_{10}}(\phi \,/\, \mathrm{Mpc^{3} \; dex^{-1}})$')
plt.show()


## FOF mass function (M200 crit)
binLimits = np.linspace(7.9, 16.9, 31)
bins = np.linspace(8.05, 16.75, 30)
lw = 3

key = 'FOF_Group_M_Crit200_DM' # 'M_DM'
phi, phi_sigma = calc_df(np.log10(dmo_pmill[key][dmo_pmill['Satellite'] == 0]), binLimits, pmill_V)
plt.plot(bins, np.log10(phi), label='P-Millennium', lw=lw)

phi, phi_sigma = calc_df(np.log10(dmo_L0100[key][dmo_L0100['Satellite'] == 0]), binLimits, 100**3)
plt.plot(bins, np.log10(phi), label='L0100N1504', lw=lw)

phi, phi_sigma = calc_df(np.log10(dmo_L0050[key][dmo_L0050['Satellite'] == 0]), binLimits, 50**3)
plt.plot(bins, np.log10(phi), label='L0050N0752', lw=lw)

plt.legend()
plt.xlim(8,)
plt.xlabel('$\mathrm{log_{10}}(M_{200} \,/\, \mathrm{M_{\odot}})$')
plt.ylabel('$\mathrm{log_{10}}(\phi \,/\, \mathrm{Mpc^{3} \; dex^{-1}})$')
plt.show()


key = 'PotentialEnergy_DM' # 'Vmax_DM' # 'velocity_DM' # 'halfMassRad_DM' # 'M_DM'
plt.hist(np.log10(np.abs(dmo_pmill[key])), histtype='step', density=True, label='PMillennium') 
plt.hist(np.log10(np.abs(dmo_L0100[key])), histtype='step', density=True, label='L0100') 
plt.hist(np.log10(np.abs(dmo_L0050[key])), histtype='step', density=True, label='L0050')
plt.legend()
plt.show()  


key = 'PotentialEnergy_DM' # 'Vmax_DM' # 'velocity_DM' # 'halfMassRad_DM' # 'M_DM'
plt.hist(np.log10(np.abs(dmo_pmill[key])), histtype='step', density=True, label='PMillennium') 
plt.hist(np.log10(np.abs(dmo_L0100[key])), histtype='step', density=True, label='L0100') 
plt.hist(np.log10(np.abs(dmo_L0050[key])), histtype='step', density=True, label='L0050')
plt.legend()
plt.show()  


