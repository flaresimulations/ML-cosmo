import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo

nthr = 4
# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')

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
                     "Subhalo/ApertureMeasurements/Mass/030kpc", 
                     numThreads=nthr, noH=True)[mask,4] * mlc.unitMass 

## Load original EAGLE AGNdT9 prediction
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
shm = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/Mass", numThreads=nthr, noH=True) * mlc.unitMass
mask = shm > 1e10
mstar_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
                            "Subhalo/ApertureMeasurements/Mass/030kpc", 
                            numThreads=nthr, noH=True)[mask,4] * mlc.unitMass 


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


## lower res Ref100 box
dmo = pd.read_csv('output/L0100N0752_%s_dmo.csv'%(mlc.tag))
dmo = dmo.loc[(dmo['M_DM'] > 1e10) & (dmo['FOF_Group_M_Crit200_DM'] > 5e9)].reset_index(drop=True)

galaxy_pred_lowres = pd.DataFrame(predictor_scaler.inverse_transform(\
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

def plot_df(ax, _mstar, binLimits, V, label, color, lw=3, ls='solid'):
    phi, phi_sigma, N = calc_df(_mstar, binLimits, V)
    N_mask = np.where(N >= 10)[0]
    N_mask_hi = np.where(N < 10)[0]
    N_mask_hi = N_mask_hi[N_mask_hi > (N_mask.max()-1)]
    N_mask_hi = np.append(N_mask_hi.min()-1,N_mask_hi)
    ax.plot(bins[N_mask], np.log10(phi[N_mask]), label=label, lw=lw, c=color, ls=ls)
    ax.plot(bins[N_mask_hi], np.log10(phi[N_mask_hi]), #label='L100Ref', 
        lw=lw, c=color, linestyle='dotted')


fig, ax = plt.subplots(1,1, figsize=(6,7))
lw = 3

plot_df(ax, np.log10(mstar), binLimits, 100**3, 'L100Ref', 'C1')
plot_df(ax, np.log10(mstar_AGNdT9), binLimits, 50**3, 'L050AGN', 'C2')

plot_df(ax, galaxy_pred_L0050['Stars_Mass_EA'], binLimits, 100**3, 
        'L050AGN\n(Prediction on L100 box)', 'C4', ls='dashed')

plot_df(ax, galaxy_pred_L0050_zoom['Stars_Mass_EA'], binLimits, 100**3, 
        'L050AGN+ZoomAGN\n(Prediction on L100 box)', 'C4')

# plot_df(ax, galaxy_pred_lowres['Stars_Mass_EA'], binLimits, 100**3, 
#         'L050AGN+ZoomAGN\n(Prediction on L100N0752 box)', 'C5')


from obs_data.baldry_12 import baldry_12

yerr = np.array([np.log10(baldry_12['phi']) - \
                    np.log10(baldry_12['phi']-baldry_12['err']),
        np.log10(baldry_12['phi']+baldry_12['err']) - \
                    np.log10(baldry_12['phi'])])

upp_limits = np.isinf(yerr)[0]
baldry_12['phi'][upp_limits] = baldry_12['phi'][upp_limits] + baldry_12['err'][upp_limits]

yerr[np.isinf(yerr)] = 0.6 # -1 * np.log10(baldry_12['phi'][np.isinf(yerr)[0]])

ax.errorbar(baldry_12['logM'], np.log10(baldry_12['phi']),
            yerr=yerr, uplims=upp_limits, color='grey', marker='o', 
            linestyle='none', label='Baldry+12', zorder=10, 
            markeredgewidth=1, markeredgecolor='black')



ax.axvspan(7, 8, alpha=0.1, color='grey')
ax.grid(alpha=0.5)
ax.set_xlim(7,13)
ax.set_ylim(-6.5,-0.8)
ax.set_xlabel('$\mathrm{log_{10}}(M_{\star} \,/\, \mathrm{M_{\odot}})$')
ax.set_ylabel('$\mathrm{log_{10}}(\phi \,/\, \mathrm{Mpc^{3} \; dex^{-1}})$')

handles, labels = ax.get_legend_handles_labels()
order = [4,0,1,2,3] # [5,0,1,2,3,4]
ax.legend(np.array(handles)[order], np.array(labels)[order], loc='lower center', ncol=2)

# plt.show()
fname = 'plots/gsmf_comparison.png' 
plt.savefig(fname, dpi=300, bbox_inches='tight') 

