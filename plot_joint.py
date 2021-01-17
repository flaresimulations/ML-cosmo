import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import seaborn as sns

from sim_details import mlcosmo
# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')

output = 'output/'
output_name = mlc.sim_name

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(output + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

train = eagle['train_mask']

galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           eagle[~train][features]))),columns=predictors)




preds = ['Stars_Mass_EA', 'MassType_Gas_EA', 'BlackHoleMass_EA',
         'StellarVelDisp_EA', 'Stars_Metallicity_EA', 'StarFormationRate_EA']
preds_pretty = ['$\mathrm{log_{10}}(M_{\star}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\mathrm{gas}}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\\bullet}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(v_{\star,\mathrm{disp}})$',
                '$\mathrm{log_{10}}(Z_{*})$',
                '$\mathrm{log_{10}}(SFR \,/\, \mathrm{M_{\odot}\, yr^{-1}})$']

ax_lims = [[8,12],[6.5,13],[4.5,10],[1.3,2.7],[-2.3,-1.4],[-3,1.2]]

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(16,9))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

plt.subplots_adjust(wspace=0.55)

for ax,pred,pretty,_lims in zip(axes, preds, preds_pretty, ax_lims):

    im = ax.hexbin(np.log10(np.ma.array(eagle[~train][pred])),
                   np.log10(np.ma.array(galaxy_pred[pred])),
                   bins='log', gridsize=20, cmap='Blues',mincnt=0,
                   extent=[_lims[0],_lims[1],_lims[0],_lims[1]])

    # cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
    cax = inset_axes(ax, width='100%', height='100%', loc=5,
                     bbox_to_anchor=[1.0, 0., 0.05, 1.0], bbox_transform=ax.transAxes)
    cbar = fig.colorbar(im, cax=cax, label='$N$')

    ax.plot([-10,15],[-10,15],linestyle='dashed',alpha=0.5, color='black')
    ax.set_xlim(_lims[0], _lims[1]);
    ax.set_ylim(_lims[0], _lims[1])
    ax.set_xlabel('%s $\, \mathrm{_{EAGLE}}$'%pretty, size=14)
    ax.set_ylabel('%s $\, \mathrm{_{Predicted}}$'%pretty, size=14)

plt.show()
# fname = 'plots/joint_plots_%s.png'%mlc.sim_name
# plt.savefig(fname, dpi=150, bbox_inches='tight')

