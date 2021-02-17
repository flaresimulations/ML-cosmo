import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error
from scipy.stats import pearsonr

from sim_details import mlcosmo
mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
# mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')

model_dir = 'models/'

fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,figsize=(18,4))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

for output_name,marker,c in zip(
        ['L0050N0752','L0050N0752_density','L0050N0752_zoom','L0050N0752_zoom_density',
         'L0100N1504','L0100N1504_density'], 
        ['o','*','o','*','o','*'], ['C0','C0','C2','C2','C1','C1']):

    etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
            pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

    train = eagle['train_mask']
    galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(\
                               etree.predict(feature_scaler.transform(\
                               eagle[~train][features]))),columns=predictors)

    pearson = []
    for p in predictors: pearson.append(round(pearsonr(eagle[~train][p],galaxy_pred[p])[0],3))

    err = pd.DataFrame({'Predictors': galaxy_pred.columns, 'Pearson': pearson})
    # scores = {}
    # for _scorer in [r2_score, explained_variance_score]:#, mean_squared_log_error]:
    #     err[_scorer.__name__] = _scorer(eagle[~train][predictors],
    #                                    galaxy_pred, multioutput='raw_values')
    for ax,prop in zip(axes, ['Stars_Mass_EA','MassType_Gas_EA','BlackHoleMass_EA', 
                              'StellarVelDisp_EA', 'StarFormationRate_EA', 'Stars_Metallicity_EA']):
        ax.scatter(np.log10(len(train)),err['Pearson'][err['Predictors']==prop],s=70,marker=marker,c=c) 
    
    for ax,label in zip(axes, ['$M_{\star}$', '$M_{\mathrm{gas}}$','$M_{\\bullet}$', 
                               '$v_{\mathrm{disp},\star}$','$\mathrm{SFR}$', '$Z_{\star}$']):
        ax.text(0.9,0.05,label,transform=ax.transAxes, horizontalalignment='right', fontsize=20)

    for ax in axes: 
        ax.set_ylim(0.5,1); ax.grid(alpha=0.5)
       
    ax3.set_xlabel('$\mathrm{log_{10}}(N_{\mathrm{halos}})$', size=14)
    ax1.set_ylabel('$\\rho_{\mathrm{pearson}}$', size=14)

    for ax in axes[1:]: ax.set_yticklabels([])


L100_patch = mpatches.Patch(color='C1', label='L100Ref')
L050_patch = mpatches.Patch(color='C0', label='L050AGN')
L050_zoom_patch = mpatches.Patch(color='C2', label='L050AGN\n+ZoomAGN')
line_ = Line2D([0], [0], color='black', marker='o', linestyle='none', label='No density')
line_dens = Line2D([0], [0], color='black', marker='*', linestyle='none', label='+ Density')
# line_zoom = Line2D([0], [0], color='black', marker='X', linestyle='none', label='Periodic+Zoom')
# line_denszoom = Line2D([0], [0], color='black', marker='D', linestyle='none', label='+Density+Zoom')
ax1.legend(handles=[L100_patch,L050_patch, L050_zoom_patch, line_, line_dens],# line_denszoom])#,
           frameon=False, loc=6, prop={'size': 13});

# plt.show()
fname = 'plots/fit_comparison.pdf'; print(fname)
plt.savefig(fname, dpi=300, bbox_inches='tight')

