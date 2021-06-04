import json
import numpy as np
import pandas as pd

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D


from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import wp# , tpcf

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo

scale_factor = 0.908563; h = 0.6777

mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
nthr = 4

fnames = [#'obs_data/farrow15-8.5-mass-9.5-2.00E-02-z-0.14-wprp.dat',
          'obs_data/farrow15-9.5-mass-10.0-2.00E-02-z-0.14-wprp.dat',  
          'obs_data/farrow15-10.0-mass-10.5-2.00E-02-z-0.14-wprp.dat',
          'obs_data/farrow15-10.5-mass-11.0-2.00E-02-z-0.14-wprp.dat',
          'obs_data/farrow15-11.0-mass-11.5-0.24-z-0.35-wprp.dat']

## P-Millennium
dmo = pd.read_csv('output/PMillennium_z000p101_dmo.csv')
# pmill_V = 800**3 # (100 / 0.6777)**3
# dmo = pd.read_csv('output/PMillennium_z000p101_dmo_subset.csv')
# pmill_V = (100 / 0.6777)**3

dmo = dmo.loc[(dmo['M_DM'] > 1e10) & (dmo['FOF_Group_M_Crit200_DM'] > 5e9)].reset_index(drop=True)
dmo['PotentialEnergy_DM'] *= 1e-2
coods_pmill = np.array(dmo[['SubPos_x','SubPos_y','SubPos_z']]) * h

mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
output_name = mlc.sim_name + '_zoom'
model_dir = 'models/'

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

galaxy_pred_L0050_zoom = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           dmo[features]))),columns=predictors)

mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
dmo = pd.read_csv('output/%s_%s_dmo.csv'%(mlc.sim_name, mlc.tag))
dmo = dmo.loc[(dmo['M_DM'] > 1e10) & (dmo['FOF_Group_M_Crit200_DM'] > 5e9)].reset_index(drop=True)
coods_l100 = np.array(dmo[['SubPos_x','SubPos_y','SubPos_z']]) * h

galaxy_pred_L100 = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           dmo[features]))),columns=predictors)

## Prediction on L100 _hydro_ features
dmo = pd.read_csv('output/%s_%s_hydro.csv'%(mlc.sim_name, mlc.tag))
dmo = dmo.loc[(dmo['M_DM'] > 1e10) & (dmo['FOF_Group_M_Crit200_DM'] > 5e9)].reset_index(drop=True)
coods_l100_hydro = np.array(dmo[['SubPos_x','SubPos_y','SubPos_z']]) * h

galaxy_pred_L100_hydro = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           dmo[features]))),columns=predictors)


rp_binlims = np.logspace(-1.5,2.1,19)
rp_bins = np.logspace(-1.4,2.0,18)


# coods_ref = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag,
#                      "Subhalo/CentreOfPotential",
#                      numThreads=nthr, noH=False, physicalUnits=False)
# 
# mstar_ref = np.log10(E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag,
#                      "Subhalo/ApertureMeasurements/Mass/030kpc",
#                      numThreads=nthr, noH=True)[:,4] * mlc.unitMass * h**2)
# 
# wp_all = {}
# wp_tiles = {}
# 
# for mstar,coods,Lbox,label,rp_max in zip(
#           [mstar_ref,galaxy_pred_L0050_zoom['Stars_Mass_EA'],
#            galaxy_pred_L100['Stars_Mass_EA']],
#           [coods_ref, coods_pmill, coods_l100],
#           [100*h, 800*h, 100*h], #100],
#           ['Ref-100','Pmill','L100'],
#           [10**1.1, 10**2, 10**1.1]): #10**1.5]):

for mstar,coods,Lbox,label,rp_max in zip([galaxy_pred_L100_hydro['Stars_Mass_EA']],
                                         [coods_l100_hydro],[100*h],
                                         ['L100_hydro'],[10**1.1]):
    
    print(label)
    wp_all[label] = {}
    wp_tiles[label] = {}

    for lim in [9.5, 10, 10.5, 11]:
        pi_max = 20.; #Lbox = 100 * h #* scale_factor

        _rp_binlims = rp_binlims[rp_binlims < rp_max]
        _rp_bins = rp_bins[:len(_rp_binlims)-1]
    
        mask = (mstar > lim) & (mstar < lim+0.5)   
        print("N_gals:", np.sum(mask))
        all_positions = return_xyz_formatted_array(coods[mask,0], coods[mask,1], coods[mask,2])

        wp_all[label][str(lim)] = wp(all_positions, _rp_binlims, pi_max, period=Lbox, 
                                num_threads='max').tolist()
    
        wp_tiles[label][str(lim)] = np.zeros((8,len(_rp_bins)))

        ## calculate jack knife errors
        for i,(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi) in \
                enumerate(zip([0, 0, Lbox/2, Lbox/2, 0, 0, Lbox/2, Lbox/2],
                    [Lbox/2, Lbox/2, Lbox, Lbox, Lbox/2, Lbox/2, Lbox, Lbox],
                    [0, Lbox/2, Lbox/2, 0, 0, Lbox/2, Lbox/2, 0],
                    [Lbox/2, Lbox, Lbox, Lbox/2, Lbox/2, Lbox, Lbox, Lbox/2],
                    [0, 0, 0, 0, Lbox/2, Lbox/2, Lbox/2, Lbox/2],
                    [Lbox/2, Lbox/2, Lbox/2, Lbox/2, Lbox, Lbox, Lbox, Lbox])):
            
            mask = (mstar > lim) & (mstar < lim+0.5)
            mask = mask & np.invert((coods[:,0] > x_lo) & (coods[:,0] < x_hi) &\
                                    (coods[:,1] > y_lo) & (coods[:,1] < y_hi) &\
                                    (coods[:,2] > z_lo) & (coods[:,2] < z_hi))
    
            all_positions = return_xyz_formatted_array(coods[mask,0], coods[mask,1], coods[mask,2])
    
            wp_tiles[label][str(lim)][i] = wp(all_positions, _rp_binlims, pi_max, 
                                         period=Lbox, num_threads='max')

        wp_tiles[label][str(lim)] = wp_tiles[label][str(lim)].tolist()
   

with open('output/clustering_wp_all.json', 'w') as outfile:
    json.dump(wp_all, outfile)

with open('output/clustering_wp_tiles.json', 'w') as outfile:
    json.dump(wp_tiles, outfile)

with open('output/clustering_wp_all.json', 'r') as outfile:
    wp_all = json.load(outfile)

with open('output/clustering_wp_tiles.json', 'r') as outfile:
    wp_tiles = json.load(outfile)

    
# fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(15,5))
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(13,10))
plt.subplots_adjust(wspace=0.03, hspace=0.03)

# ax2B = inset_axes(ax2, width='40%', height='30%', loc=3)
ax2B = zoomed_inset_axes(ax2, zoom=2.3, loc=3)

for ax,lim,fname in zip([ax1,ax2,ax3,ax4], ['9.5', '10', '10.5', '11'], fnames):
    for Lbox,label,pretty_label,c,rp_max in zip([100*h, 800*h, 100*h, 100*h],
                             ['Ref-100','Pmill','L100','L100_hydro'],
                             ['L100Ref','L050AGN+Zoom\n(Prediction on\nP-Millennium)',
                               'L050AGN+Zoom\n(Prediction on\nL100)','L100 Hydro'], 
                             ['C0','C1','C3','C4'],
                             [10**1.1, 10**2, 10**1.1, 10**1.1]):
        
        if (label in ['L100','L100_hydro']) & (lim != '10'): continue
        if (label in ['Ref-100']) & (lim == '11'): continue
        
        _rp_binlims = rp_binlims[rp_binlims < rp_max]
        _rp_bins = rp_bins[:len(_rp_binlims)-1]

        sigma = np.sqrt(np.sum((np.array(wp_all[label][lim]) - \
                                np.array(wp_tiles[label][lim]))**2, axis=0) \
                * (len(np.array(wp_tiles[label][lim])) - 1)/\
                len(np.array(wp_tiles[label][lim]))) / _rp_bins
    
        _y = np.array(wp_all[label][lim]) / _rp_bins
        err = np.array([np.log10(_y) - np.log10(_y - sigma), np.log10(_y + sigma) - np.log10(_y)])
        uplims = np.isnan(err[0])
        err[np.isnan(err)] = 0.5

        ax.errorbar(np.log10(_rp_bins), np.log10(wp_all[label][lim] / _rp_bins),
                    yerr=err, capsize=2, c=c)
        ax.errorbar(np.log10(_rp_bins), np.log10(wp_all[label][lim] / _rp_bins),
                    yerr=err, label=pretty_label, capsize=2, uplims=uplims, c=c)
        
        if lim=='10':
            ax2B.errorbar(np.log10(_rp_bins), np.log10(wp_all[label][lim] / _rp_bins), 
                          yerr=err,capsize=2, c=c)
            ax2B.errorbar(np.log10(_rp_bins), np.log10(wp_all[label][lim] / _rp_bins),
                          yerr=err, label=pretty_label, capsize=2, uplims=uplims, c=c)


    ## obs data    
    _dat = np.loadtxt(fname)
    ax.fill_between(np.log10(_dat[:,0]), np.log10((_dat[:,1]-_dat[:,2])/_dat[:,0]), 
                                         np.log10((_dat[:,1]+_dat[:,2])/_dat[:,0]), 
                                         alpha=0.5, color='grey')

    ax.plot(np.log10(_dat[:,0]), np.log10(_dat[:,1]/_dat[:,0]), label='GAMA', color='black')
    if lim == '10': 
        ax2B.fill_between(np.log10(_dat[:,0]), np.log10((_dat[:,1]-_dat[:,2])/_dat[:,0]), 
                          np.log10((_dat[:,1]+_dat[:,2])/_dat[:,0]), 
                          alpha=0.5, color='grey')

        ax2B.plot(np.log10(_dat[:,0]), np.log10(_dat[:,1]/_dat[:,0]), 
                            label='GAMA', color='black')
   
    ax.text(0.93, 0.92, '$%.1f < \mathrm{log_{10}}(M_{\star} / M_{\odot} h^{-2}) < %.1f$'%\
            (float(lim),float(lim)+0.5), transform=ax.transAxes, size=13, horizontalalignment='right')
    ax.set_xlim(-1.7,2); ax.set_ylim(-3.5,5)
    ax.grid(alpha=0.4)

 

ax2B.set_xlim(-1.42, -0.75); ax2B.set_ylim(2.8, 4.7)
ax2B.grid(alpha=0.4)
mark_inset(ax2, ax2B, loc1=2, loc2=1, fc="none", ec="0.5")

for ax in [ax2,ax4,ax2B]: ax.set_yticklabels([])
for ax in [ax1,ax2,ax2B]: ax.set_xticklabels([])
for ax in [ax1,ax3]: ax.set_ylabel('$\mathrm{log_{10}}(w_{p}(r_{p})\,/\,r_{p})$', size=13)
for ax in [ax3,ax4]: ax.set_xlabel('$\mathrm{log_{10}}(r_{p} \,/\, h^{-1} \mathrm{Mpc})$', size=13)


lineGAMA = Line2D([0], [0], color='black')
lineA = Line2D([0], [0], color='C0')
lineB = Line2D([0], [0], color='C1')
lineC = Line2D([0], [0], color='C3')
lineD = Line2D([0], [0], color='C4')
ax1.legend([lineGAMA, lineA,lineB], 
           ['GAMA','L100Ref','L050AGN+Zoom\n(Prediction on\nP-Millennium)'],loc='lower left')
ax2.legend([lineC, lineD], 
           ['L050AGN+Zoom\n(Prediction on\nL100 DMO)','L050AGN+Zoom\n(Prediction on\nL100 hydro)'], 
           loc=(0.63,0.58))

plt.show()
# plt.savefig('plots/clustering.png', dpi=200, bbox_inches='tight')
