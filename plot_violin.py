import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import seaborn as sns

from sim_details import mlcosmo
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')

output = 'output/'
galaxy_pred = pd.read_csv('%sprediction_%s_%s.csv'%(output,mlc.sim_name,mlc.tag))
eagle = pd.read_csv('%sfeatures_%s_%s.csv'%(output,mlc.sim_name,mlc.tag))

train = eagle['train_mask']

def vplot_data(feature):
    vdata = pd.DataFrame(np.log10(np.ma.array(galaxy_pred[feature])), columns=[feature])
    vdata['type'] = 'prediction'

    vdata_temp = pd.DataFrame(np.log10(np.ma.array(eagle[~train][feature])),  columns=[feature])
    vdata_temp['type'] = 'eagle'

    vdata = vdata.append(vdata_temp)
    vdata['dummy'] = 'A'
    return vdata


# fig = plt.figure(figsize=(16,16))
# gs = gridspec.GridSpec(2,13)
# ax1 = fig.add_subplot(gs[0,0:4])
# ax2 = fig.add_subplot(gs[0,4:8])
# ax3 = fig.add_subplot(gs[0,8:12])
# ax4 = fig.add_subplot(gs[1,0:4])
# ax5 = fig.add_subplot(gs[1,4:8])
# ax6 = fig.add_subplot(gs[1,8:12])

fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,figsize=(16,6))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

preds = ['Stars_Mass_EA', 'MassType_Gas_EA', 'BlackHoleMass_EA',
         'StellarVelDisp_EA', 'Stars_Metallicity_EA', 'StarFormationRate_EA']
preds_pretty = ['$\mathrm{log_{10}}(M_{\star}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\mathrm{gas}}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(M_{\\bullet}\,/\,\mathrm{M_{\odot}})$',
                '$\mathrm{log_{10}}(v_{\star,\mathrm{disp}})$',
                '$\mathrm{log_{10}}(Z_{*})$',
                '$\mathrm{log_{10}}(SFR \,/\, \mathrm{M_{\odot}\, yr^{-1}})$']
ax_lims = [[7,12],[5,12],[4,9.5],[1,2.5],[-2.5,-1.5],[-4.5,1]]

for ax,pred,pretty,_lims in zip(axes, preds, preds_pretty, ax_lims):

    vdata = vplot_data(pred)
    sns.violinplot(x='dummy', y=pred, hue='type',
                   split=True, data=vdata, palette="Set2",
                   inner='quartile', ax=ax, cut=0)

    ax.legend_.remove()
    ax.set_ylabel('')
    ax.set_title(pretty, fontsize=14)
    ax.set_ylim(_lims[0],_lims[1])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_visible(False)


ax1.legend(loc=4, fontsize=13, frameon=False)#bbox_to_anchor=(1.3, 0.95), 

# plt.subplots_adjust(wspace=1.5)
plt.show()
# fname = 'plots/violins_%s.png'%mlc.sim_name
# plt.savefig(fname, dpi=150, bbox_inches='tight')

