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
zoom = True
density = False
output = 'models/'
output_name = mlc.sim_name 
if zoom: output_name += '_zoom'
if density: output_name += '_density'

etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(output + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

train = eagle['train_mask']

galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(\
                           etree.predict(feature_scaler.transform(\
                           eagle[~train][features]))),columns=predictors)

# mask = np.array(galaxy_pred['Stars_Mass_EA'] > -1)
# galaxy_pred = galaxy_pred[mask]
# eagle = eagle[~train][mask]

from operator import le, ge

def vplot_data(feature,lim=-5,op=ge):   
    vdata_temp = pd.DataFrame(np.ma.array(eagle[~train][feature]),  columns=[feature])

    mask = op(vdata_temp['StarFormationRate_EA'], lim)

    vdata_temp = vdata_temp.loc[mask].reset_index(drop=True)
    vdata_temp['type'] = 'eagle'
    
    vdata = pd.DataFrame(np.ma.array(galaxy_pred[feature]), columns=[feature])
    vdata = vdata.loc[mask].reset_index(drop=True)
    vdata_temp['type'] = 'eagle'
    vdata['type'] = 'prediction'

    vdata = vdata.append(vdata_temp)
    vdata['dummy'] = 'A'
    return vdata


# fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,6))

fig = plt.figure(constrained_layout=True,figsize=(7,4.5))
widths = [5,5,3.2]
spec = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths)
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])
ax3 = fig.add_subplot(spec[2])
axes = [ax1,ax2,ax3]

ax_lims = [-3.5,0.8]
sfr_lims = [-5,-3.1,-3.1] 
operators = [ge,ge,le]

for i,(ax,sfrlim,_op) in enumerate(zip(axes, sfr_lims, operators)):

    vdata = vplot_data('StarFormationRate_EA', sfrlim, op=_op)
    if i==2:
        sns.violinplot(x='dummy', y='StarFormationRate_EA', hue='type', split=False, 
                       data=vdata.loc[vdata['type'] == 'prediction'], 
                   palette="Set2", inner='quartile', ax=ax, cut=0)
    else:
        sns.violinplot(x='dummy', y='StarFormationRate_EA', hue='type', split=True, data=vdata, 
                       palette="Set2", inner='quartile', ax=ax, cut=0)

    ax.legend_.remove()
    ax.set_ylabel('')
    # ax.set_title('$\mathrm{log_{10}}(SFR \,/\, \mathrm{M_{\odot}\, yr^{-1}})$', fontsize=14)
    ax.set_ylim(ax_lims[0],ax_lims[1])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_visible(False)

ax1.set_ylabel('$\mathrm{log_{10}}(SFR \,/\, \mathrm{M_{\odot}\, yr^{-1}})$', fontsize=14)
ax1.legend(loc=(0.1,-0.18), fontsize=13, frameon=False)
ax2.text(0.24, -0.1, '$\\frac{\mathrm{SFR_{eagle}}}{\mathrm{M_{\odot}\, yr^{-1}}} > 10^{-3}$', 
         size=14, transform=ax2.transAxes)
ax3.text(0.1, -0.1, '$\\frac{\mathrm{SFR_{eagle}}}{\mathrm{M_{\odot}\, yr^{-1}}} < 10^{-3}$', 
         size=14, transform=ax3.transAxes)

plt.show()
# fname = 'plots/violins_SFR_%s.png'%mlc.sim_name; print(fname)
# plt.savefig(fname, dpi=150, bbox_inches='tight')

