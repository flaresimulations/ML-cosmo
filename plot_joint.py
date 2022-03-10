import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes


from sim_details import mlcosmo
# # mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
mlc = mlcosmo(ini='config/config_flares.ini')

model_dir = 'models/'
output_name = 'flares' # mlc.sim_name
# zoom = True #False
# density = True #False
# output_name = mlc.sim_name
# if zoom: output_name += '_zoom'
# if density: output_name += '_density'


etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

train = eagle['train_mask']

galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           eagle[~train][features]))),columns=predictors)

# mask = np.array(galaxy_pred['Stars_Mass_EA'] > 1e8)

preds = ['Stars_Mass_EA', 'MassType_Gas_EA', 'BlackHoleMass_EA',
         'StellarVelDisp_EA', 'StarFormationRate_EA', 'Stars_Metallicity_EA']
preds_pretty = ['$\mathrm{log_{10}}(M_{\star}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\mathrm{gas}}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\\bullet}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(v_{\star,\mathrm{disp}})$',
                '$\mathrm{log_{10}}(SFR \,/\, \mathrm{M_{\odot}\, yr^{-1}})$',
                '$\mathrm{log_{10}}(Z_{*})$']

## fractions of galaxies within x dex
diff = {pred: None for pred in preds}
for pred in preds:
    diff[pred] = np.array(eagle[~train][pred]) - \
                 np.array(galaxy_pred[pred])

    print(pred, "%0.3f"%(np.sum(diff[pred] < 0.2) / len(diff[pred])))
    
    mask = eagle[~train]['Stars_Mass_EA'] > 9
    print(pred, "%0.3f"%(np.sum(diff[pred][mask] < 0.2) / len(diff[pred][mask])))



ax_lims = [[4.5,13],[5.4,11],[4,10],[0.3,3],[-4.6,2.5],[-6.8,-1]]

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(16,9))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

plt.subplots_adjust(wspace=0.55)

for ax,pred,pretty,_lims in zip(axes, preds, preds_pretty, ax_lims):

    print(pred, pretty, _lims)
    im = ax.hexbin(np.ma.array(eagle[~train][pred]),
                   np.ma.array(galaxy_pred[pred]),
                   bins='log', gridsize=20, cmap='Blues',mincnt=0,
                   extent=[_lims[0],_lims[1],_lims[0],_lims[1]])
    
    # im = ax.hexbin(np.log10(np.ma.array(eagle[~train][~mask][pred])),
    #                np.log10(np.ma.array(galaxy_pred[~mask][pred])),
    #                bins='log', gridsize=20, cmap='Reds',mincnt=10, alpha=0.5,
    #                extent=[_lims[0],_lims[1],_lims[0],_lims[1]])

    # cax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
    cax = inset_axes(ax, width='100%', height='100%', loc=5,
                     bbox_to_anchor=[1.0, 0., 0.05, 1.0], bbox_transform=ax.transAxes)
    cbar = fig.colorbar(im, cax=cax, label='$N$')

    ax.plot([-10,15],[-10,15],linestyle='dashed',alpha=0.5, color='black')
    ax.set_xlim(_lims[0], _lims[1]);
    ax.set_ylim(_lims[0], _lims[1])
    ax.set_xlabel('%s $\, \mathrm{_{EAGLE}}$'%pretty, size=14)
    ax.set_ylabel('%s $\, \mathrm{_{Predicted}}$'%pretty, size=14)
    ax.text(0.05, 0.88, "Percentage of predictions\nwithin 0.2 dex: %0.0f%s"%\
            (100* np.sum(diff[pred] < 0.2) / len(diff[pred]), chr(37)), 
            transform=ax.transAxes)


# plt.show()
# fname = 'plots/joint_plots_%s.pdf'%mlc.sim_name; print(fname)
fname = 'plots/joint_plots_%s.pdf'%'flares'; print(fname)
plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()


