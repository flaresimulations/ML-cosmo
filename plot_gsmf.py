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

nthr = 4
# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')

V = 100**3 # Mpc^3
output = 'output/'


## Load DMO simulation
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
dmo = pd.read_csv('output/%s_%s_dmo.csv'%(mlc.sim_name, mlc.tag))
dmo = dmo.loc[(dmo['M_DM'] > 1e10) & (dmo['FOF_Group_M_Crit200_DM'] > 5e9)].reset_index(drop=True)

## Load original EAGLE ref prediction
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
shm = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Mass", numThreads=nthr, noH=True) * mlc.unitMass
mask = shm > 1e10
mstar = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Stars/Mass", numThreads=nthr, noH=True)[mask] * mlc.unitMass

## Load original EAGLE AGNdT9 prediction
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
shm = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Mass", numThreads=nthr, noH=True) * mlc.unitMass
mask = shm > 1e10
mstar_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Stars/Mass", numThreads=nthr, noH=True)[mask] * mlc.unitMass

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
    return phi, phi_sigma, hist


binLimits = np.linspace(4.9, 13.9, 31)
bins = np.linspace(5.05, 13.75, 30)

fig, ax = plt.subplots(1,1, figsize=(6,7))

lw = 3

def plot_df(ax, mstar, binLimits, V, label, color, lw=3, ls='solid'):
    phi, phi_sigma, N = calc_df(mstar, binLimits, V)
    N_mask = np.where(N >= 10)[0]
    N_mask_hi = np.where(N < 10)[0]
    print(N_mask)
    print(N_mask_hi)
    N_mask_hi = N_mask_hi[N_mask_hi > (N_mask.max()-1)]
    N_mask_hi = np.append(N_mask_hi.min()-1,N_mask_hi)
    print(N_mask)
    print(N_mask_hi)

    ax.plot(bins[N_mask], np.log10(phi[N_mask]), label=label, lw=lw, c=color, ls=ls)
    ax.plot(bins[N_mask_hi], np.log10(phi[N_mask_hi]), #label='L100Ref', 
        lw=lw, c=color, linestyle='dotted')

plot_df(ax, np.log10(mstar), binLimits, 100**3, 'L100Ref', 'C1')
plot_df(ax, np.log10(mstar_AGNdT9), binLimits, 50**3, 'L050AGN', 'C2')

plot_df(ax, galaxy_pred_L0050['Stars_Mass_EA'], binLimits, 100**3, 
        'L050AGN\n(Prediction on L100 box)', 'C4', ls='dashed')

plot_df(ax, galaxy_pred_L0050_zoom['Stars_Mass_EA'], binLimits, 100**3, 
        'L050AGN+ZoomAGN\n(Prediction on L100 box)', 'C4')

# phi, phi_sigma, N = calc_df(np.log10(mstar_AGNdT9), binLimits, 50**3)
# ax.plot(bins, np.log10(phi), label='L050AGN', lw=lw, c='C2')

# phi, phi_sigma, N = calc_df(galaxy_pred_L0050['Stars_Mass_EA'], 
#                          binLimits, 100**3)
# ax.plot(bins, np.log10(phi), label='L050AGN\n(Prediction on L100 box)', lw=lw, c='C4', linestyle='dashed')

# phi, phi_sigma, N = calc_df(galaxy_pred_L0050_zoom['Stars_Mass_EA'], 
#                         binLimits, 100**3)
# ax.plot(bins, np.log10(phi), label='L050AGN+ZoomAGN\n(Prediction on L100 box)', lw=lw, c='C4')

from obs_data.baldry_12 import baldry_12

yerr = np.array([np.log10(baldry_12['phi']) - \
                    np.log10(baldry_12['phi']-baldry_12['err']),
        np.log10(baldry_12['phi']+baldry_12['err']) - \
                    np.log10(baldry_12['phi'])])

upp_limits = np.isinf(yerr)[0]
baldry_12['phi'][upp_limits] = baldry_12['phi'][upp_limits] + baldry_12['err'][upp_limits]

yerr[np.isinf(yerr)] = 0.6 # -1 * np.log10(baldry_12['phi'][np.isinf(yerr)[0]])

ax.errorbar(baldry_12['logM'], np.log10(baldry_12['phi']),
            yerr=yerr, uplims=upp_limits, color='grey', marker='o', linestyle='none')


ax.axvspan(7, 8, alpha=0.1, color='grey')
ax.grid(alpha=0.5)
ax.legend(loc='lower center')
ax.set_xlim(7,13)
ax.set_ylim(-8,-0.5)
ax.set_xlabel('$\mathrm{log_{10}}(M_{\star} \,/\, \mathrm{M_{\odot}})$')
ax.set_ylabel('$\mathrm{log_{10}}(\phi \,/\, \mathrm{Mpc^{3} \; dex^{-1}})$')

# plt.show()
fname = 'plots/gsmf_comparison.png' 
plt.savefig(fname, dpi=300, bbox_inches='tight') 

