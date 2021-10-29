import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from scipy.stats import pearsonr

from sim_details import mlcosmo
# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')

model_dir = 'models/'
zoom = True #False
density = False
output_name = mlc.sim_name
if zoom: output_name += '_zoom'
if density: output_name += '_density'

# _shmass = {}
# _shmass['M_DM'] = pd.DataFrame(np.logspace(9,16,200), columns=['M_DM'])
# _shmass['Vmax_DM'] = pd.DataFrame(np.logspace(1,4,200), columns=['Vmax_DM'])
galaxy_pred = {}
train = {}

output_name_new = output_name + '_sham_mass'
etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name_new + '_' + mlc.tag + '_ert.model', 'rb'))

train['M_DM'] = eagle['train_mask']
galaxy_pred['M_DM'] = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                  pd.DataFrame(eagle['M_DM'],columns=['M_DM'])[~train['M_DM']] ))),
                           columns=predictors)

output_name_new = output_name + '_sham_vmax'
etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name_new + '_' + mlc.tag + '_ert.model', 'rb'))

train['Vmax_DM'] = eagle['train_mask']
galaxy_pred['Vmax_DM'] = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                 pd.DataFrame(eagle['Vmax_DM'],columns=['Vmax_DM'])[~train['Vmax_DM']] ))),
                           columns=predictors)

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

_train = eagle['train_mask']

galaxy_pred_all = pd.DataFrame(predictor_scaler.inverse_transform(\
                               etree.predict(feature_scaler.transform(\
                               eagle[~_train][features]))),columns=predictors)




from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds="clip")

# galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(\
#                            etree.predict(feature_scaler.transform(\
#                            eagle[~train][features]))),columns=predictors)

# train = eagle['train_mask']


preds = ['Stars_Mass_EA', 'MassType_Gas_EA', 'BlackHoleMass_EA',
         'StellarVelDisp_EA', 'StarFormationRate_EA', 'Stars_Metallicity_EA']
preds_pretty = ['$\mathrm{log_{10}}(M_{\star}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\mathrm{gas}}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\\bullet}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(v_{\star,\mathrm{disp}})$',
                '$\mathrm{log_{10}}(SFR \,/\, \mathrm{M_{\odot}\, yr^{-1}})$',
                '$\mathrm{log_{10}}(Z_{*})$']


## fit subhalo and stellar mass relation

fig, all_axes = plt.subplots(6,2, figsize=(8,14))
axesa,axesb = all_axes[:,0], all_axes[:,1]

_pearson = {}
_pearson_all = {}
for feat, xlabel, _x, _xlim_lo, _xlim_hi, axes in \
                              zip(['M_DM', 'Vmax_DM'], 
                                  ['$\mathrm{log_{10}}(M_{\mathrm{subhalo}} \,/\, M_{\odot})$', 
                                   '$\mathrm{log_{10}}(V_{\mathrm{max}} \,/\, \mathrm{km \; s^{-1}})$'],
                                  [np.linspace(10,16), np.linspace(1,4)],
                                  [10,1.3], [15,3.4], [axesa,axesb]):
    
    _pearson[feat] = {}
    for ax, pred, _ylim_lo, _ylim_hi in zip(axes, preds, [4.9,5,4,0,-5,-10], [15,14,12.5,3.8,4,4]):

        # _sort = np.argsort(eagle[feat][~train[feat]])
        # ax.plot(np.log10(eagle[feat][~train[feat]].iloc[_sort]), galaxy_pred[feat][pred].iloc[_sort], 
        #         color='black', linestyle='dashed')
    
        ir.fit(np.log10(eagle[feat][_train]), eagle[pred][_train])
        _y = ir.predict(_x)
        ax.plot(_x,_y, color='black', zorder=10)

        # p = np.polyfit(np.log10(eagle[feat]), eagle[pred], deg=1)
        # print(pred, len(p))
        # _y = p[0]*_x + p[1]
        # # _y = p[0]*_x**2 + p[1]*_x + p[2]
        # ax.plot(_x, _y, color='black', zorder=10)
        
        _new_x = np.log10(eagle[feat])
        # new_pred = p[0]*_new_x + p[1]
        # new_pred = p[0]*_x**2 + p[1]*_x + p[2]
        
        ax.hexbin(np.log10(eagle[feat]), eagle[pred],
                           bins='log', gridsize=30, cmap='Blues',mincnt=0,
                           extent=[_xlim_lo,_xlim_hi,_ylim_lo,_ylim_hi])
       
        new_pred = ir.predict(_new_x)
        diff = np.array(eagle[pred]) - np.array(new_pred)
        _pearson[feat][pred] = round(pearsonr(eagle[pred],new_pred)[0],3)
        _pearson_all[pred] = round(pearsonr(eagle[pred][~_train], galaxy_pred_all[pred])[0],3)

        # diff = np.array(eagle[pred][~train[feat]]) - np.array(galaxy_pred[feat][pred])
        ax.text(0.05, 0.78, "Percentage of predictions\nwithin 0.2 dex: %0.0f%s"%\
                (np.sum(diff < 0.2) * 100 / len(diff), chr(37)), transform=ax.transAxes)

        ax.set_xlim(_xlim_lo,_xlim_hi)
        ax.set_ylim(_ylim_lo,_ylim_hi)
        

    axes[-1].set_xlabel(xlabel)

for ax, label in zip(axesa, preds_pretty): ax.set_ylabel(label)
plt.subplots_adjust(wspace=0.2)

plt.show()
# fname = 'plots/sham_comparison_%s.pdf'%mlc.sim_name; print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()




## Pearson plot comparison
fig,ax = plt.subplots(1,1, figsize=(6.2,5))

_s = 100
ax.scatter(np.arange(6), _pearson['M_DM'].values(), s=_s, color='C5',
           label='Isotonic model ($M_{\mathrm{sub}})$')

ax.scatter(np.arange(6), _pearson['Vmax_DM'].values(), s=_s, color='C5', marker='X',
           label='Isotonic model ($V_{\mathrm{max}}$)')

ax.scatter(np.arange(6), _pearson_all.values(), s=_s, color='C2',
           label='ERT (L050AGN+Zoom)', zorder=0)

ax.set_xticks(np.arange(6))
ax.set_xticklabels(preds_pretty, rotation=90)

ax.set_ylim(0.4,1.0)
ax.set_xlim(-0.5,5.5)
ax.legend(ncol=2, bbox_to_anchor=(0.1, 1.1))
# ax.grid(alpha=0.5)
ax.vlines(np.arange(6)-0.5, 0,1, color='black')

ax.set_ylabel('$\\rho_{\mathrm{pearson}}$')

plt.show()
# fname = 'plots/pearson_isotonic_comparison.pdf'; print(fname)
# plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close()

