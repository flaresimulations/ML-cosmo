import glob
import pandas as pd
import numpy as np
import pickle 

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score

from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import seaborn as sns

from sim_details import mlcosmo
# mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
output = 'output/'
model_dir = 'models/'

zoom = False
density = True
# output_name = mlc.sim_name + '_zoom' + '_density'
output_name = mlc.sim_name + '_density'

eagle = pd.read_csv((output + mlc.sim_name + '_' + mlc.tag + "_match.csv"))

if zoom:
    files = glob.glob(output+'CE*_029_z000p000_match.csv')
    for f in files:
        _dat = pd.read_csv(f)
        eagle = pd.concat([eagle,_dat])


eagle['velocity_DM'] = abs(eagle['velocity_DM'])

features = ['FOF_Group_M_Crit200_DM',
            'FOF_Group_R_Crit200_DM',
            'M_DM', #'Subhalo_Mass_DM',
            'MassTwiceHalfMassRad_DM',
            'velocity_DM',
            'Vmax_DM',
            'VmaxRadius_DM',
            'Satellite']

if density:
    [features.append(d) for d in ['Density_R1','Density_R2','Density_R4','Density_R8','Density_R16']]


features_pretty = ['$M_{\mathrm{Crit,200}}$',
                   '$R_{\mathrm{Crit,200}}$',
                   '$M_{\mathrm{subhalo}} \,/\, M_{\odot}$',
                   '$M_{2 \\times R\, 1/2} \,/\, M_{\odot}$',
                   '$v$',
                   '$v_{\mathrm{max}}$',
                   '$R_{v_{\mathrm{max}}}$',
                   'Satellite?',
                   '$\\rho (R = 1 \, \mathrm{Mpc})$',
                   '$\\rho (R = 2 \, \mathrm{Mpc})$',
                   '$\\rho (R = 4 \, \mathrm{Mpc})$',
                   '$\\rho (R = 8 \, \mathrm{Mpc})$']

predictors = [
              # 'M_EA','lengthType_BH_EA',
              'BlackHoleMass_EA','BlackHoleMassAccretionRate_EA',
              'GasSpin_EA',
              'halfMassProjRad_Gas_EA','halfMassProjRad_Stars_EA','halfMassProjRad_BH_EA',
              'halfMassRad_Gas_EA','halfMassRad_Stars_EA','halfMassRad_BH_EA',
              'KineticEnergy_EA',
              #'InitialMassWeightedBirthZ_EA',
              #'InitialMassWeightedStellarAge_EA',
              'MassType_Gas_EA','MassType_Stars_EA','MassType_BH_EA',
              'MassTwiceHalfMassRad_Gas_EA','MassTwiceHalfMassRad_Stars_EA',
              'MassTwiceHalfMassRad_BH_EA',
              #'StellarInitialMass_EA',
              'Stars_Mass_EA',
              'Stars_Spin_EA',
              'Stars_Metallicity_EA',
              'StellarVelDisp_EA',
              'StellarVelDisp_HalfMassProjRad_EA',
              'StarFormationRate_EA']



mask = (eagle['M_DM'] > 1e9) & (eagle['Stars_Mass_EA'] > 1e8)
print("N:",np.sum(mask),"\nN(excluded galaxies):",np.sum(~mask))
eagle = eagle[mask]


split = 0.8
train = np.random.rand(len(eagle)) < split
feature_scaler = preprocessing.StandardScaler().fit(eagle[train][features])
predictor_scaler = preprocessing.StandardScaler().fit(eagle[train][predictors])

## ---- Cross Validation

ss = KFold(n_splits=10, shuffle=True)
tuned_parameters = {'n_estimators': [10,15],#,25,50], 
                    'min_samples_split': [10,20]}#,10,20]}#, 'max_features': ['auto','sqrt','log2'] }

etree = GridSearchCV(ExtraTreesRegressor(), param_grid=tuned_parameters, cv=None, n_jobs=3)


etree.fit(feature_scaler.transform(eagle[train][features]), predictor_scaler.transform(eagle[train][predictors]))

print(etree.best_params_)

eagle['train_mask'] = train

pickle.dump([etree, features, predictors, feature_scaler, predictor_scaler, eagle], 
            open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'wb'))


# ## ---- Prediction
# galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(etree.predict(feature_scaler.transform(eagle[~train][features]))),columns=predictors)
# 
# 
# 
# galaxy_pred.to_csv('%sprediction_%s_%s.csv'%(output,mlc.sim_name,mlc.tag))
# eagle[predictors].to_csv('%sfeatures_%s_%s.csv'%(output,mlc.sim_name,mlc.tag))

# ## ---- Feature importance
# 
# importance_etree = etree.best_estimator_.feature_importances_
# idx = importance_etree.argsort()[::-1]
# sorted_features = np.asarray(features_pretty)[idx]
# 
# pos = np.arange(len(idx))
# plt.bar(pos,importance_etree[idx], align='center')
# plt.xticks(pos, sorted_features, rotation='vertical')
# plt.ylabel('Importance')
# # plt.xlabel('Feature')
# plt.show()
# # fname = 'plots/feature_importance_%s.png'%mlc.sim_name
# # plt.savefig(fname, dpi=150, bbox_inches='tight')
# 
# 
# 
# ## ---- Errors
# r2_ert = r2_score(eagle[~train][predictors], galaxy_pred, multioutput='raw_values')
# 
# pearson = []
# for p in predictors:
#     pearson.append(round(pearsonr(eagle[~train][p],galaxy_pred[p])[0],3))
# 
# 
# err = pd.DataFrame({'Predictors': galaxy_pred.columns,'R2': r2_ert.round(3), 'Pearson': pearson})
# # err[['Predictors','R2','Pearson']].sort_values(by='R2',ascending=False)
# 
