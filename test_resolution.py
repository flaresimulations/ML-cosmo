import sys
import glob

import pandas as pd
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
eagle['Density_R1'] *= mlc.unitMass
eagle['Density_R2'] *= mlc.unitMass
eagle['Density_R4'] *= mlc.unitMass
eagle['Density_R8'] *= mlc.unitMass
# eagle['Density_R16'] *= mlc.unitMass


if zoom:
    if density:
        files = [output+'CE%i_029_z000p000_match.csv'%i for i in np.arange(8)]
    else:
        files = glob.glob(output+'CE*_029_z000p000_match.csv')

    for f in files:
        _dat = pd.read_csv(f)
        eagle = pd.concat([eagle,_dat])


bins = np.linspace(0,14,100) 
lims = [1e5,1e10,2e10,4e10,8e10] # 2e11
labels = ['All',
          '$M_{\,\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}} > 1 \\times 10^{10}$',  
          '$M_{\,\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}} > 2 \\times 10^{10}$',
          '$M_{\,\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}} > 4 \\times 10^{10}$',  
          '$M_{\,\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}} > 8 \\times 10^{10}$']
#,'$2 \\times 10^{11}$'])):  #'$6 \\times 10^{10}$',

fig = plt.figure(figsize=(7,9.5))
spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, hspace=0.05)
ax1 = fig.add_subplot(spec[1:, 0])
ax2 = fig.add_subplot(spec[0, 0])

for i,(lim,label) in enumerate(zip(lims,labels)):
    _temp = eagle[eagle['M_DM'] > lim]['Stars_Mass_EA'].copy() 
    _temp[_temp == 0] = 1e5
    n,dummy,dummy = ax1.hist(np.log10(_temp), histtype='step', bins=bins, label=label, 
             lw=2, log=True, color=matplotlib.cm.plasma(i/len(lims))) 
    if i == 0:
        N = n
    ax2.plot(bins[1:] + (bins[1]-bins[0])/2, n/N, color=matplotlib.cm.plasma(i/len(lims)))
 

for ax in [ax1,ax2]:
    ax.set_xlim(4.9,12.5) 
    # ax.vlines(np.log10(1.8e8),0,1, linestyle='dashed')
    ax.axvspan(1, np.log10(1.8e8), alpha=0.1, color='grey')

ax1.set_ylim(1,3.1e5) 
ax2.set_ylim(0,1)
ax2.set_xticklabels([])
ax.grid(alpha=0.3,axis='y')
ax1.legend()#title='$M_{\,\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}}$')
ax1.set_xlabel('$M_{\star} \,/\, \mathrm{M_{\odot}}$',size=14)
ax1.set_ylabel('$N$',size=14)
ax2.set_ylabel('$\mathrm{Completeness}$',size=14)

# plt.show() 
fname = 'plots/stellar_mass_completeness.pdf'; print(fname)
plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close() 
