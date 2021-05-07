import numpy as np
import pandas as pd
import pickle
import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo


output = 'output/'
nthr=2
Z_sol = 0.0127
## Load DMO simulation

# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
# dmo = pd.read_csv('output/%s_%s_dmo.csv'%(mlc.sim_name, mlc.tag))


# dmo = pd.read_csv('output/PMillennium_z000p101_dmo.csv')
# pmill_V = 800**3 # (100 / 0.6777)**3
dmo = pd.read_csv('output/PMillennium_z000p101_dmo_subset.csv')
pmill_V = 100**3 # (100 / 0.6777)**3
dmo = dmo[(dmo['FOF_Group_M_Crit200_DM'] > 5e9) & (dmo['M_DM'] > 1e10)].reset_index()



## Load original EAGLE ref prediction
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
shm = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass", noH=True) * mlc.unitMass
central = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)==0
mask = (shm > 1e10) & central
mstar = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/ApertureMeasurements/Mass/030kpc", noH=True)[mask,4] * mlc.unitMass
mbh = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMass", noH=True)[mask] * mlc.unitMass
sfr = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StarFormationRate", noH=True)[mask]
met = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Metallicity", noH=True)[mask] / Z_sol


## Load original EAGLE AGNdT9
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
shm_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass", noH=True) * mlc.unitMass
central = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)==0
mask = (shm_AGNdT9 > 1e10) & central
mstar_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/ApertureMeasurements/Mass/030kpc", noH=True)[mask,4]*mlc.unitMass
mbh_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMass", noH=True)[mask]*mlc.unitMass
sfr_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StarFormationRate", noH=True)[mask]
met_AGNdT9 = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Metallicity", noH=True)[mask] / Z_sol

# load zoom + L050 AGNdT9 
zoomL050 = pd.read_csv((output + mlc.sim_name + '_' + mlc.tag + "_match.csv"))
files = glob.glob(output+'CE*_026_z000p101_match.csv')
for f in files:
    _dat = pd.read_csv(f)
    zoomL050 = pd.concat([zoomL050,_dat])

zoomL050  = zoomL050[(zoomL050['M_DM'] > 1e10) &\
                     (zoomL050['FOF_Group_M_Crit200_DM'] > 5e9) &\
                     (zoomL050['Satellite'] == 0)]


# sfr[sfr == 0.] = 1e-4
# sfr_AGNdT9[sfr_AGNdT9 == 0.] = 1e-4
# zoomL050.loc[zoomL050['StarFormationRate_EA'] == 0.,'StarFormationRate_EA'] = 1e-4
# met[met == 0.] = 1e-5
# met_AGNdT9[met_AGNdT9 == 0.] = 1e-5
# zoomL050.loc[zoomL050['Stars_Metallicity_EA'] == 0.,'Stars_Metallicity_EA'] = 1e-4


## Load predictions
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
output_name = mlc.sim_name + '_zoom'
model_dir = 'models/'

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

galaxy_pred_L0050_zoom = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           dmo[features][dmo['Satellite'] == 0]))),columns=predictors)


def calc_df(x, binLimits, volume):
    hist, dummy = np.histogram(x, bins = binLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (binLimits[1] - binLimits[0])
    phi_sigma = (np.sqrt(hist) / volume) /\
                (binLimits[1] - binLimits[0]) # Poisson errors
    return phi, phi_sigma, hist



fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18,7))
lw = 3; axes = [ax1,ax2,ax3]
binLimits = np.linspace(4.9, 13.9, 31); bins = np.linspace(5.05, 13.75, 30)

_alpha = 0.2

## BH-stellar mass relation
percentiles, sigma, bin_mask = mlc.find_percs(np.log10(mstar), np.log10(mbh), binLimits)
ax1.plot(bins, percentiles[1], label='L100Ref', lw=lw, c='C1')
ax1.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C1')

percentiles, sigma, bin_mask = mlc.find_percs(np.log10(mstar_AGNdT9), np.log10(mbh_AGNdT9), binLimits)
ax1.plot(bins, percentiles[1], label='L050AGN', lw=lw, c='C2')
ax1.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C2')

percentiles, sigma, bin_mask = mlc.find_percs(np.log10(zoomL050['Stars_Mass_EA']), 
                           np.log10(zoomL050['BlackHoleMass_EA']), binLimits)
ax1.plot(bins, percentiles[1], label='L050AGN+ZoomAGN', lw=lw, c='C3')
ax1.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C3')

percentiles, sigma, bin_mask = mlc.find_percs(galaxy_pred_L0050_zoom['Stars_Mass_EA'],
                                              galaxy_pred_L0050_zoom['BlackHoleMass_EA'], binLimits)
ax1.plot(bins, percentiles[1], label='L050AGN+ZoomAGN\n(Prediction on P-Millennium)', lw=lw, c='C0')
ax1.fill_between(bins, percentiles[0], percentiles[2], alpha=_alpha, color='C0')

mcconnel_ma_12 = pd.read_csv('obs_data/mcconnell_ma_12.txt', delim_whitespace=True)
_errs = [np.log10(mcconnel_ma_12['M_BH']) - np.log10(mcconnel_ma_12['M_BH'] - mcconnel_ma_12['M_BH_lo']),
         np.log10(mcconnel_ma_12['M_BH'] + mcconnel_ma_12['M_BH_hi']) - np.log10(mcconnel_ma_12['M_BH'])]
ax1.errorbar(np.log10(mcconnel_ma_12['M_bulge']), np.log10(mcconnel_ma_12['M_BH']), yerr=_errs, 
             color='grey', marker='o', linestyle='none', label='McConnel & Ma+13', zorder=10,
             markeredgewidth=1, markeredgecolor='black')




## metallity-stellar mass relation
percentiles, sigma, bin_mask = mlc.find_percs(np.log10(mstar), np.log10(met), binLimits)
ax2.plot(bins, percentiles[1], lw=lw, c='C1')# label='Ref-100',
ax2.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C1')

percentiles, sigma, bin_mask = mlc.find_percs(np.log10(mstar_AGNdT9), np.log10(met_AGNdT9), binLimits)
ax2.plot(bins, percentiles[1], lw=lw, c='C2')# label='AGNdT9-50',
ax2.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C2')

percentiles, sigma, bin_mask = mlc.find_percs(np.log10(zoomL050['Stars_Mass_EA']), 
                           np.log10(zoomL050['Stars_Metallicity_EA'] / Z_sol), binLimits)
ax2.plot(bins, percentiles[1], lw=lw, c='C3')#label='AGNdT9-50', 
ax2.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C3')

percentiles, sigma, bin_mask = mlc.find_percs(galaxy_pred_L0050_zoom['Stars_Mass_EA'],
               np.log10(10**galaxy_pred_L0050_zoom['Stars_Metallicity_EA'] / Z_sol), binLimits)
ax2.plot(bins, percentiles[1],lw=lw, c='C0')# label='Zoom-pred', 
ax2.fill_between(bins, percentiles[0], percentiles[2], alpha=_alpha, color='C0')

from obs_data.galazzi_05 import galazzi_05
_errs = [galazzi_05['Z_p50'] - galazzi_05['Z_p16'], galazzi_05['Z_p84'] - galazzi_05['Z_p50']]
ax2.errorbar(galazzi_05['Mstar'], galazzi_05['Z_p50'], yerr=_errs,
             color='grey', marker='o', linestyle='none', label='Galazzi+05', zorder=10,
             markeredgewidth=1, markeredgecolor='black')

## star forming sequence
ssfr_lim = 1e-11
mask = (sfr / mstar) > ssfr_lim
percentiles, sigma, bin_mask = mlc.find_percs(np.log10(mstar[mask]), np.log10(sfr[mask]), binLimits)
ax3.plot(bins, percentiles[1], lw=lw, c='C1')# label='Ref-100',
ax3.fill_between(bins, percentiles[0], percentiles[2], alpha=_alpha, color='C1')

mask = (sfr_AGNdT9 / mstar_AGNdT9) > ssfr_lim
percentiles, sigma, bin_mask = mlc.find_percs(np.log10(mstar_AGNdT9[mask]), np.log10(sfr_AGNdT9[mask]), binLimits)
ax3.plot(bins, percentiles[1], lw=lw, c='C2')# label='AGNdT9-50',
ax3.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C2')

mask = (zoomL050['StarFormationRate_EA'] / zoomL050['Stars_Mass_EA']) > ssfr_lim
percentiles, sigma, bin_mask = mlc.find_percs(np.log10(zoomL050['Stars_Mass_EA'][mask]), 
                           np.log10(zoomL050['StarFormationRate_EA'][mask]), binLimits)
ax3.plot(bins, percentiles[1], lw=lw, c='C3')#label='AGNdT9-50+Zoom', 
ax3.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C3')

mask = (galaxy_pred_L0050_zoom['StarFormationRate_EA'] -\
        galaxy_pred_L0050_zoom['Stars_Mass_EA']) > np.log10(ssfr_lim)
percentiles, sigma, bin_mask = mlc.find_percs(
                   galaxy_pred_L0050_zoom['Stars_Mass_EA'][mask],
                   galaxy_pred_L0050_zoom['StarFormationRate_EA'][mask], binLimits)
ax3.plot(bins, percentiles[1], lw=lw, c='C0')# label='Zoom-pred',
ax3.fill_between(bins[bin_mask], percentiles[0][bin_mask], percentiles[2][bin_mask], 
                 alpha=_alpha, color='C0')

x = np.linspace(8,13)
ax3.plot(x, x+np.log10(ssfr_lim), linestyle='dashed', color='black')

_obs = pd.read_csv('obs_data/bauer_13.txt')#, delim_whitespace=True)
_y = np.log10((10**_obs['sSFR']) * (10**_obs['M_star']) * 1e-9)
_errs = [_y - np.log10((10**_obs['sSFR_lo']) * (10**_obs['M_star']) * 1e-9),
         np.log10((10**_obs['sSFR_hi']) * (10**_obs['M_star']) * 1e-9) - _y]
ax3.errorbar(_obs['M_star'], _y, yerr=_errs,
             color='grey', marker='o', linestyle='none', label='Bauer+13', zorder=10,
             markeredgewidth=1, markeredgecolor='black')


for ax in axes:
    ax.legend()
    ax.grid(alpha=0.5)
    ax.set_xlim(8,13)
    ax.set_xlabel('$\mathrm{log_{10}}(M_{\star} \,/\, \mathrm{M_{\odot}})$', size=14)

ax1.set_ylim(5,)
ax3.set_ylim(-3,2.2)
ax2.set_ylim(-1.1,0.7)

ax1.set_ylabel('$M_{\\bullet} \,/\, \mathrm{M_{\odot}}$', size=14)
ax3.set_ylabel('$\mathrm{SFR} \,/\, \mathrm{M_{\odot} \, yr^{-1}}$', size=14)
ax2.set_ylabel('$Z_{\star}$', size=14)

plt.show()
# fname = 'plots/pmillennium_dfs.png'; plt.savefig(fname, dpi=250, bbox_inches='tight')
# fname = 'plots/L0100_dfs.png'; plt.savefig(fname, dpi=250, bbox_inches='tight')
