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


# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')

V = 100**3 # Mpc^3
output = 'output/'


## Load DMO simulation
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
dmo = pd.read_csv('output/%s_%s_dmo.csv'%(mlc.sim_name, mlc.tag))

## Load original EAGLE ref prediction
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


binLimits = np.linspace(7.9, 13.9, 21)
bins = np.linspace(8.05, 13.75, 20)
lw = 3

phi, phi_sigma = calc_df(np.log10(mstar), binLimits, 100**3)
plt.plot(bins, np.log10(phi), label='True (Ref-100)', lw=lw)

phi, phi_sigma = calc_df(np.log10(mstar_AGNdT9), binLimits, 50**3)
plt.plot(bins, np.log10(phi), label='True (AGNdT9-50)', lw=lw)

phi, phi_sigma = calc_df(np.log10(galaxy_pred_L0050['Stars_Mass_EA']), 
                         binLimits, 100**3)
plt.plot(bins, np.log10(phi), label='L0050 Prediction', lw=lw)

phi, phi_sigma= calc_df(np.log10(galaxy_pred_L0050_zoom['Stars_Mass_EA']), 
                        binLimits, 100**3)
plt.plot(bins, np.log10(phi), label='L0050-zoom Prediction', lw=lw)

plt.legend()
plt.xlim(8,)
plt.xlabel('$\mathrm{log_{10}}(M_{\star} \,/\, \mathrm{M_{\odot}})$')
plt.ylabel('$\mathrm{log_{10}}(\phi \,/\, \mathrm{Mpc^{3} \; dex^{-1}})$')
plt.show()

