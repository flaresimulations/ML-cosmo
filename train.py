import sys
import glob

import pandas as pd
import numpy as np
import pickle 

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse

output = 'output/'
model_dir = 'models/'

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file", type=str)
parser.add_argument("--tag", help="snapshot tag string", type=str, default=None)
parser.add_argument("--density", help="download local density features", action='store_true')
args = parser.parse_args()

from sim_details import mlcosmo
mlc = mlcosmo(ini=args.config, tag=args.tag)


# from sim_details import mlcosmo
# mlc = mlcosmo(ini=sys.argv[1])
# # mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')


# zoom = bool(int(sys.argv[2]))
# density = bool(int(sys.argv[3]))
output_name = 'flares' #mlc.sim_name
if args.density: output_name += '_density'
print(output_name)

# eagle = pd.read_csv((output + mlc.sim_name + '_' + mlc.tag + "_match.csv"))
# eagle['Density_R1'] *= mlc.unitMass
# eagle['Density_R2'] *= mlc.unitMass
# eagle['Density_R4'] *= mlc.unitMass
# eagle['Density_R8'] *= mlc.unitMass
# eagle['Density_R16'] *= mlc.unitMass

# if args.zoom:
    # if density: 
    #     files = [output+'CE%i_029_z000p000_match.csv'%i for i in np.arange(8)]
    # else: 
    # files = glob.glob(output+'CE*_%s_match.csv'%mlc.tag)
    # files = glob.glob(output+'CE*_026_z000p101_match.csv')

files = glob.glob(f'{output}FL*_{mlc.tag}_match.csv')

for i,f in enumerate(files):
    _dat = pd.read_csv(f)
    if i==0: eagle = _dat
    eagle = pd.concat([eagle,_dat])



eagle['velocity_DM'] = abs(eagle['velocity_DM'])

features = ['FOF_Group_M_Crit200_DM',
            'M_DM',
            'halfMassRad_DM',
            'velocity_DM',
            'Vmax_DM',
            'PotentialEnergy_DM',
            'VmaxRadius_DM',
            'Satellite']

features_pretty = ['$M_{\mathrm{Crit,200}}$',
                   '$M_{\mathrm{subhalo}} \,/\, M_{\odot}$',
                   '$R_{1/2 \; \mathrm{mass}}$',
                   '$v$',
                   '$v_{\mathrm{max}}$',
                   '$E_{\mathrm{pot}}$',
                   '$R_{v_{\mathrm{max}}}$',
                   'Satellite?']
if args.density:
    [features.append(d) for d in ['Density_R1','Density_R2','Density_R4','Density_R8']]#,'Density_R16']]
    [features_pretty.append(d) for d in ['$\\rho (R = 1 \, \mathrm{Mpc})$',
                                         '$\\rho (R = 2 \, \mathrm{Mpc})$',
                                         '$\\rho (R = 4 \, \mathrm{Mpc})$',
                                         '$\\rho (R = 8 \, \mathrm{Mpc})$']]#,
                                         #'$\\rho (R = 16 \, \mathrm{Mpc})$']]

predictors = [
              # 'M_EA','lengthType_BH_EA',
              'BlackHoleMass_EA',
              # 'BlackHoleMassAccretionRate_EA',
              # 'GasSpin_EA',
              #'halfMassProjRad_Gas_EA','halfMassProjRad_Stars_EA','halfMassProjRad_BH_EA',
              # 'halfMassRad_Gas_EA','halfMassRad_Stars_EA','halfMassRad_BH_EA',
              # 'KineticEnergy_EA',
              #'InitialMassWeightedBirthZ_EA',
              #'InitialMassWeightedStellarAge_EA',
              'MassType_Gas_EA',
              #'MassType_Stars_EA','MassType_BH_EA',
              # 'MassTwiceHalfMassRad_Gas_EA','MassTwiceHalfMassRad_Stars_EA',
              # 'MassTwiceHalfMassRad_BH_EA',
              #'StellarInitialMass_EA',
              'Stars_Mass_EA',
              'Stars_Spin_EA',
              'Stars_Metallicity_EA',
              'StellarVelDisp_EA',
              # 'StellarVelDisp_HalfMassProjRad_EA',
              'StarFormationRate_EA']


mask = (eagle['M_DM'] > 1e10) & (eagle['FOF_Group_M_Crit200_DM'] > 5e9)

print("N:",np.sum(mask),"\nN(excluded galaxies):",np.sum(~mask))
eagle = eagle[mask]

# set zeros to small values
eagle.loc[eagle['Stars_Mass_EA'] == 0.,'Stars_Mass_EA'] = 1e5
eagle.loc[eagle['MassType_Gas_EA'] == 0.,'MassType_Gas_EA'] = 5e5
eagle.loc[eagle['BlackHoleMass_EA'] == 0.,'BlackHoleMass_EA'] = 2e4
eagle.loc[eagle['StellarVelDisp_EA'] == 0.,'StellarVelDisp_EA'] = 3
eagle.loc[eagle['Stars_Metallicity_EA'] == 0.,'Stars_Metallicity_EA'] = 5e-7
eagle.loc[eagle['StarFormationRate_EA'] == 0.,'StarFormationRate_EA'] = 1e-4

eagle['MassType_Gas_EA'] = np.log10(eagle['MassType_Gas_EA'])
eagle['Stars_Mass_EA'] = np.log10(eagle['Stars_Mass_EA'])
eagle['BlackHoleMass_EA'] = np.log10(eagle['BlackHoleMass_EA'])
eagle['Stars_Metallicity_EA'] = np.log10(eagle['Stars_Metallicity_EA'])
eagle['StellarVelDisp_EA'] = np.log10(eagle['StellarVelDisp_EA'])
eagle['StarFormationRate_EA'] = np.log10(eagle['StarFormationRate_EA'])

# for i in [1,2,4,8]:
#     eagle['Density_R%i'%i] = np.log10(eagle['Density_R%i'%i])

# eagle['FOF_Group_M_Crit200_DM'] = np.log10(eagle['FOF_Group_M_Crit200_DM'])
# eagle['M_DM'] = np.log10(eagle['M_DM'])
# eagle['halfMassRad_DM'] = np.log10(eagle['halfMassRad_DM'])
# # eagle['velocity_DM'] = np.log10(eagle['velocity_DM'])
# # eagle['Vmax_DM'] = np.log10(eagle['Vmax_DM'])
# eagle['PotentialEnergy_DM'] = np.log10(eagle['PotentialEnergy_DM'])
# # eagle['VmaxRadius_DM'] = np.log10(eagle['VmaxRadius_DM'])

## Extra features
# eagle['halo_mass_ratio'] = np.log10(eagle['M_DM'] / eagle['FOF_Group_M_Crit200_DM'])
# 
# features.append('halo_mass_ratio')
# features_pretty.append('Halo mass ratio')

split = 0.8
train = np.random.rand(len(eagle)) < split
print("train / test:", np.sum(train), np.sum(~train))

feature_scaler = preprocessing.StandardScaler().fit(eagle[train][features])
predictor_scaler = preprocessing.StandardScaler().fit(eagle[train][predictors])
# feature_scaler = preprocessing.RobustScaler().fit(eagle[train][features])
# predictor_scaler = preprocessing.RobustScaler().fit(eagle[train][predictors])

## ---- Cross Validation

ss = KFold(n_splits=5, shuffle=True)
tuned_parameters = {
                    'n_estimators': [25,35,45,55],
                    'min_samples_split': [5,15,25,35], 
                    'min_samples_leaf': [2,4,6,8], 
                    }

etree = GridSearchCV(ExtraTreesRegressor(), param_grid=tuned_parameters, cv=None, n_jobs=4)

etree.fit(feature_scaler.transform(eagle[train][features]), predictor_scaler.transform(eagle[train][predictors]))

print(etree.best_params_)

# regr = MLPRegressor(random_state=1, max_iter=700, verbose=True, tol=1e-5)
# regr.fit(feature_scaler.transform(eagle[train][features]), predictor_scaler.transform(eagle[train][predictors]))

eagle['train_mask'] = train



# pickle.dump([regr, features, predictors, feature_scaler, predictor_scaler, eagle], 
pickle.dump([etree, features, predictors, feature_scaler, predictor_scaler, eagle], 
            open(model_dir + output_name + '_' + mlc.tag + '_ert.model', 'wb'))


## ---- Prediction
galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(etree.predict(feature_scaler.transform(eagle[~train][features]))),columns=predictors)
# galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(regr.predict(feature_scaler.transform(eagle[~train][features]))),columns=predictors)

# galaxy_pred.to_csv('%sprediction_%s_%s.csv'%(output,mlc.sim_name,mlc.tag))
# eagle[predictors].to_csv('%sfeatures_%s_%s.csv'%(output,mlc.sim_name,mlc.tag))

## ---- Errors
# r2_ert = r2_score(eagle[~train][predictors], galaxy_pred, multioutput='raw_values')

pearson = []
for p in predictors:
    pearson.append(round(pearsonr(eagle[~train][p],galaxy_pred[p])[0],3))


err = pd.DataFrame({'Predictors': galaxy_pred.columns, 'Pearson': pearson})

scores = {}
for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
    err[_scorer.__name__] = _scorer(eagle[~train][predictors], 
                                       galaxy_pred, multioutput='raw_values')


print(err.sort_values(by='r2_score',ascending=False))

# ## ---- Feature importance
importance_etree = etree.best_estimator_.feature_importances_
idx = importance_etree.argsort()[::-1]
sorted_features = np.asarray(features_pretty)[idx]

pos = np.arange(len(idx))
plt.bar(pos,importance_etree[idx], align='center')
plt.xticks(pos, sorted_features, rotation='vertical')
plt.ylabel('Importance')
# plt.xlabel('Feature')
plt.show()
# # # fname = 'plots/feature_importance_%s.png'%mlc.sim_name
# # # plt.savefig(fname, dpi=150, bbox_inches='tight')


