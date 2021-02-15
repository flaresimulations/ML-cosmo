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

dmo = pd.read_csv('output/PMillennium_z000p000_dmo.csv')
# dmo = pd.read_csv('output/PMillennium_z000p000_dmo_subset.csv')
# dmo = dmo[dmo['FOF_Group_M_Crit200_DM'] > 2e9].reset_index()
dmo = dmo[dmo['M_DM'] > 1e9].reset_index()
pmill_V = 800**3 # (100 / 0.6777)**3
# dmo['PotentialEnergy_DM'] *= 1e-3


## Load original EAGLE ref prediction
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
mstar = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Stars/Mass") * mlc.unitMass

## Load original EAGLE AGNdT9 prediction
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
mstar_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Stars/Mass") * mlc.unitMass

## Load predictions

mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
output_name = mlc.sim_name # + '_zoom'
model_dir = 'models/'

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

galaxy_pred_L0050 = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           dmo[features]))),columns=predictors)

####

mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
output_name = mlc.sim_name + '_zoom'
model_dir = 'models/'

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

galaxy_pred_L0050_zoom = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           dmo[features]))),columns=predictors)


def calc_df(x, binLimits, volume):
    hist, dummy = np.histogram(x, bins = binLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (binLimits[1] - binLimits[0])
    phi_sigma = (np.sqrt(hist) / volume) /\
                (binLimits[1] - binLimits[0]) # Poisson errors
    return phi, phi_sigma


binLimits = np.linspace(7.9, 16.9, 31)
bins = np.linspace(8.05, 16.75, 30)

phi_pred, phi_sigma = calc_df(np.log10(galaxy_pred_L0050['Stars_Mass_EA']), 
                         binLimits, pmill_V)

phi_pred_zoom, phi_sigma= calc_df(np.log10(galaxy_pred_L0050_zoom['Stars_Mass_EA']), 
                        binLimits, pmill_V)


fig, ax = plt.subplots(1,1, figsize=(6,7))
lw = 3

phi, phi_sigma = calc_df(np.log10(mstar), binLimits, 100**3)
ax.plot(bins, np.log10(phi), label='Ref-100', lw=lw, c='C1')

phi, phi_sigma = calc_df(np.log10(mstar_AGNdT9), binLimits, 50**3)
ax.plot(bins, np.log10(phi), label='AGNdT9-50', lw=lw, c='C2')

ax.plot(bins, np.log10(phi_pred), label='P-Millennium Prediction', lw=lw, c='C0', linestyle='dashed')
ax.plot(bins, np.log10(phi_pred_zoom), label='P-Millennium Prediction (+zoom)', lw=lw, c='C0')

ax.axvspan(8, 9, alpha=0.1, color='grey')

ax.legend()
ax.grid(alpha=0.5)
ax.set_xlim(8,13)
ax.set_ylim(-6.5,0.5)
ax.set_xlabel('$\mathrm{log_{10}}(M_{\star} \,/\, \mathrm{M_{\odot}})$')
ax.set_ylabel('$\mathrm{log_{10}}(\phi \,/\, \mathrm{Mpc^{3} \; dex^{-1}})$')

# plt.show()
fname = 'plots/gsmf_pmillennium.png'
plt.savefig(fname, dpi=150, bbox_inches='tight')

