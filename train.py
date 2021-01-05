import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


eagle = pd.read_csv('output/028_z000p000_match.csv')

eagle['velocity_DM'] = abs(eagle['velocity_DM'])
eagle['Subhalo_Mass_DM'] = 1.15*10**7 * eagle['length_DM']


features = ['FOF_Group_M_Mean200_DM','FOF_Group_R_Mean200_DM',
            'Subhalo_Mass_DM','velocity_DM','Vmax_DM']

predictors = ['M_EA','lengthType_BH_EA','BlackHoleMass_EA','BlackHoleMassAccretionRate_EA',
              'GasSpin_EA','halfMassProjRad_Gas_EA','halfMassProjRad_Stars_EA',
              'halfMassProjRad_BH_EA','halfMassRad_Gas_EA','halfMassRad_Stars_EA',
              'halfMassRad_BH_EA','KineticEnergy_EA','InitialMassWeightedBirthZ_EA',
              'InitialMassWeightedStellarAge_EA','MassType_Gas_EA','MassType_Stars_EA',
              'MassType_BH_EA','MassTwiceHalfMassRad_Gas_EA','MassTwiceHalfMassRad_Stars_EA',
              'MassTwiceHalfMassRad_BH_EA','StellarInitialMass_EA','Stars_Mass_EA','Stars_Spin_EA',
              'Stars_Metallicity_EA','StellarVelDisp_EA','StellarVelDisp_HalfMassProjRad_EA',
              'StarFormationRate_EA']



mask = (eagle['FOF_Group_M_Mean200_DM'] > 1e9)# & (eagle['MassType_Stars_EA'] > 0.)
eagle = eagle[mask]

print("N(excluded galaxies):",sum(mask==False))


split = 0.8
train = np.random.rand(len(eagle)) < split



feature_scaler = preprocessing.StandardScaler().fit(eagle[train][features])
print(feature_scaler.mean_)
print(feature_scaler.scale_)


predictor_scaler = preprocessing.StandardScaler().fit(eagle[train][predictors])
print(predictor_scaler.mean_)
print(predictor_scaler.scale_)


## ---- Cross Validation

ss = KFold(n_splits=10, shuffle=True)
tuned_parameters = {'n_estimators': [50], 'min_samples_split': [5]}#, 'max_features': ['auto','sqrt','log2'] }
etree = GridSearchCV(ExtraTreesRegressor(), param_grid=tuned_parameters, cv=None, n_jobs=3)


etree.fit(feature_scaler.transform(eagle[train][features]), predictor_scaler.transform(eagle[train][predictors]))

print(etree.best_params_)

## ---- Prediction

galaxy_pred = pd.DataFrame(predictor_scaler.inverse_transform(etree.predict(feature_scaler.transform(eagle[~train][features]))),columns=predictors)




## ---- Errors
r2_ert = r2_score(eagle[~train][predictors], galaxy_pred, multioutput='raw_values')


pearson = []
for p in predictors:
    pearson.append(round(pearsonr(eagle[~train][p],galaxy_pred[p])[0],3))


err = pd.DataFrame({'Predictors': galaxy_pred.columns,'R2': r2_ert.round(3), 'Pearson': pearson})
# err[['Predictors','R2','Pearson']].sort_values(by='R2',ascending=False)


## ---- Violin Plot
def vplot_data(feature):
    vdata = pd.DataFrame(np.log10(np.ma.array(galaxy_pred[feature])), columns=[feature])
    vdata['type'] = 'prediction'

    vdata_temp = pd.DataFrame(np.log10(np.ma.array(eagle[~train][feature])),  columns=[feature])
    vdata_temp['type'] = 'eagle'

    vdata = vdata.append(vdata_temp)
    vdata['dummy'] = 'A'
    return vdata


fig = plt.figure(figsize=(16,16))

gs = gridspec.GridSpec(2,13)

ax1 = fig.add_subplot(gs[0,0:4])
ax2 = fig.add_subplot(gs[0,4:8])
ax3 = fig.add_subplot(gs[0,8:12])
ax4 = fig.add_subplot(gs[1,0:4])
ax5 = fig.add_subplot(gs[1,4:8])
ax6 = fig.add_subplot(gs[1,8:12])

axes = [ax1,ax2,ax3,ax4,ax5,ax6]
preds = ['Stars_Mass_EA', 'MassType_Gas_EA', 'BlackHoleMass_EA', 'StellarVelDisp_EA', 'Stars_Metallicity_EA', 'StarFormationRate_EA']
preds_pretty = ['$\\rm{log}(M_{*}/M_{\odot})$', '$\\rm{log}(M_{gas}/M_{\odot})$', '$M_{BH}/M_{\odot}$', '$v_{*,disp}$', '$Z_{*}$', '$SFR (M_{*} yr^{-1})$']

for ax,pred,pretty in zip(axes,preds, preds_pretty):

    vdata = vplot_data(pred)
    sns.violinplot(x='dummy', y=pred, hue='type', split=True, data=vdata, palette="Set2",
                   inner='quartile', ax=ax)
    ax.legend_.remove()
    ax.set_ylabel('')
    ax.set_title(pretty, fontsize=18)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_visible(False)

ax6.legend(bbox_to_anchor=(1.3, 0.95), fontsize=13)

# fig.tight_layout()
# plt.subplots_adjust(wspace=1.5)
plt.show()


## ---- Joint plots



sns.set_style("ticks")
sns.set_context("poster")

def typical_plot_calls(g):
    #g.ax_marg_x.set_axis_off()
    #g.ax_marg_y.set_axis_off()
    cax = g.fig.add_axes([.815, 0.125, .03, .689])
    plt.colorbar(cax=cax, label='$\mathrm{log}_{10}(N)$')
    ax = g.ax_joint
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: r"$10^{{{}}}$".format(int(x))))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: r"$10^{{{}}}$".format(int(x))))
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')




#mask = eagle[~train]['Stars_Mass_EA'] > 1e7 #& (galaxy_pred['Stars_Mass_EA'] > 1e7)

g = sns.JointGrid(np.log10(np.ma.array(eagle[~train]['Stars_Mass_EA'])),
                  np.log10(np.ma.array(galaxy_pred['Stars_Mass_EA'])),
                  space=0, xlim=(7.01, 11.7), ylim=(7.01,11.7))

g.plot_joint(plt.hexbin, bins='log', gridsize=60, cmap="Blues")
g.set_axis_labels('$(M_{*} / M_{\odot}) \\rm{_{EAGLE}}$','$(M_{*} / M_{\odot}) \\rm{_{Predicted}}$')
typical_plot_calls(g)

plt.show()

#plt.savefig('../../output/stellarmass_predicted_jointplot.png', bbox_inches='tight', dpi=300)

## ---- Feature importance

importance_etree = etree.best_estimator_.feature_importances_
idx = importance_etree.argsort()[::-1]
sorted_features = np.asarray(features[0:5])[idx]



pos = np.arange(len(idx))
plt.bar(pos,importance_etree[idx], align='center')
plt.xticks(pos, sorted_features, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.show()
# plt.style.use("seaborn-white")


