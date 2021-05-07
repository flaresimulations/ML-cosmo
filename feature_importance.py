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

from sim_details import mlcosmo
mlc = mlcosmo(ini=sys.argv[1])
# mlc = mlcosmo(ini='config/config_cosma_L0050N0752.ini')
output = 'output/'
model_dir = 'models/'


zoom = bool(int(sys.argv[2]))
density = bool(int(sys.argv[3]))
output_name = mlc.sim_name
if zoom: output_name += '_zoom'
if density: output_name += '_density'
print(output_name)

eagle = pd.read_csv((output + mlc.sim_name + '_' + mlc.tag + "_match.csv"))
# eagle['Density_R1'] *= mlc.unitMass
# eagle['Density_R2'] *= mlc.unitMass
# eagle['Density_R4'] *= mlc.unitMass
# eagle['Density_R8'] *= mlc.unitMass
# eagle['Density_R16'] *= mlc.unitMass


if zoom:
    # if density: 
    #     files = [output+'CE%i_029_z000p000_match.csv'%i for i in np.arange(8)]
    # else: 
    # files = glob.glob(output+'CE*_%s_match.csv'%mlc.tag)
    files = glob.glob(output+'CE*_026_z000p101_match.csv')

    for f in files:
        _dat = pd.read_csv(f)
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
if density:
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
              #'Stars_Spin_EA',
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


split = 0.8
train = np.random.rand(len(eagle)) < split
print("train / test:", np.sum(train), np.sum(~train))
eagle['train_mask'] = train

feature_scaler = preprocessing.StandardScaler().fit(eagle[train][features])
predictor_scaler = preprocessing.StandardScaler().fit(eagle[train][predictors])


ss = KFold(n_splits=5, shuffle=True)
tuned_parameters = {
                    'n_estimators': [25],#,35,45,55],
                    'min_samples_split': [15],#[5,15,25,35], 
                    'min_samples_leaf': [8], # [2,4,6,8], 
                    }

etree = GridSearchCV(ExtraTreesRegressor(), param_grid=tuned_parameters, cv=None, n_jobs=4)


_out = {predictor: {feature: None for feature in features} for predictor in predictors}

for i,predictor in enumerate(predictors):
    print(predictor)
    etree.fit(feature_scaler.transform(eagle[train][features]), 
            predictor_scaler.transform(eagle[train][predictors])[:,i])
    
    importance_etree = etree.best_estimator_.feature_importances_
    
    for j,feature in enumerate(features):
        _out[predictor][feature] = importance_etree[j]



predictors_pretty = ['$M_{\cdot} \,/\, M_{\odot}$',
                     '$M_{\mathrm{gas}} \,/\, M_{\odot}$',
                     '$M_{\star} \,/\, M_{\odot}$',
                     '$Z_{\star}$',
                     '$v_{\mathrm{disp,\star}}$',
                     '$\mathrm{SFR} \,/\, M_{\odot} \, \mathrm{yr^{-1}}$',
                    ]

H = np.array([[_out[predictor][feature] for feature in features] for predictor in predictors])
_H = H.T / H.max(axis=1)

fig, ax = plt.subplots(1,1,figsize=(5, 6))
plt.imshow(_H)

pos = np.arange(len(idx))
plt.yticks(np.arange(len(features)), features_pretty)#, rotation='vertical')
plt.xticks(np.arange(len(predictors)), predictors_pretty, rotation='vertical')

cax = fig.add_axes([0.14, 0.12, 0.88, 0.75])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
cbar = plt.colorbar(orientation='vertical')
cbar.set_label('Relative feature importance', rotation=270, labelpad=12)

plt.show()
# fname = 'plots/feature_importance_predictors_%s.png'%mlc.sim_name
# plt.savefig(fname, dpi=250, bbox_inches='tight')

