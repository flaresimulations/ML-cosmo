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

features = ['FOF_Group_M_Crit200_DM', 'M_DM', 'halfMassRad_DM', 'velocity_DM', 'Vmax_DM',
            'PotentialEnergy_DM', 'VmaxRadius_DM', 'Satellite']

features_pretty = ['$M_{\mathrm{Crit,200}}$','$M_{\mathrm{subhalo}} \,/\, M_{\odot}$',
                   '$R_{1/2 \; \mathrm{mass}}$','$v$','$v_{\mathrm{max}}$','$E_{\mathrm{pot}}$',
                   '$R_{v_{\mathrm{max}}}$','Satellite?']


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,8))
plt.subplots_adjust(hspace=0.5)
dx = 0.2

for i,(output_name,name,c) in enumerate(zip(
        ['L0050N0752','L0050N0752_zoom','L0100N1504'],
        ['$\mathrm{L050AGN}$','$\mathrm{L050AGN+Zoom}$','$\mathrm{L100Ref}$'],
        # 'L0100N1504_density'],'L0050N0752_density','L0050N0752_zoom_density'
        ['C0','C1','C2','C3'])):

    etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
            pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

    ## ---- Feature importance
    importance_etree = etree.best_estimator_.feature_importances_
    idx = importance_etree.argsort()[::-1]
    sorted_features = np.asarray(features_pretty)[idx]
    
    pos = np.arange(len(idx))+ i*dx
    ax1.bar(pos,importance_etree[idx], align='center', color=c, width=0.2, label=name)


ax1.set_xticks(pos-0.2)
ax1.set_xticklabels(sorted_features, rotation='vertical')
ax1.set_ylabel('Importance')
ax1.legend()


[features.append(d) for d in ['Density_R1','Density_R2','Density_R4','Density_R8']]
[features_pretty.append(d) for d in ['$\\rho (R = 1 \, \mathrm{Mpc})$',
                                     '$\\rho (R = 2 \, \mathrm{Mpc})$',
                                     '$\\rho (R = 4 \, \mathrm{Mpc})$',
                                     '$\\rho (R = 8 \, \mathrm{Mpc})$']]

for i,(output_name,name,c) in enumerate(zip(
        ['L0050N0752_density','L0050N0752_zoom_density'],
        ['$\mathrm{L050AGN + Density}$','$\mathrm{L050AGN+Zoom + Density}$'],
        ['C0','C1'])):
    etree, features, predictors, feature_scaler, predictor_scaler, eagle =\
        pickle.load(open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'rb'))

    ## ---- Feature importance
    importance_etree = etree.best_estimator_.feature_importances_
    idx = importance_etree.argsort()[::-1]
    sorted_features = np.asarray(features_pretty)[idx]
    
    pos = np.arange(len(idx))+ i*dx
    ax2.bar(pos,importance_etree[idx], align='center', color=c, width=0.2, label=name)

ax2.set_xticks(pos)
ax2.set_xticklabels(sorted_features, rotation='vertical')
ax2.set_ylabel('Importance')
ax2.legend()

# plt.show()
fname = 'plots/feature_importance_all.png'
plt.savefig(fname, dpi=300, bbox_inches='tight')
