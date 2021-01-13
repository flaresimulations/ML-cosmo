import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo
_config = str(sys.argv[1])
mlc = mlcosmo(ini=_config)

output = 'output/'

eagle_match = pd.read_csv(output + mlc.sim_name + '_' + mlc.tag + '_match.csv')
indexes = np.loadtxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt', dtype=int)

eagle_mass = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Mass") * 1e10

unmatched = ~np.in1d(range(len(eagle_mass)), indexes[:,0]) 
# unmatched_dmo = ~pd.Series(range(len(eagle_dmo))).isin(pickle_data['idx_DM'])

## Matched fraction
m_limit = 1e9
print(np.sum(eagle_mass[indexes[:,0]] > m_limit) / np.sum(eagle_mass > m_limit))


## Histogram of subhalo masses
massBinLimits = np.linspace(5.2, 15.4, 52)
massBins = np.logspace(5.3, 15.3, 51)
binWidths = []
for i,z in enumerate(massBins):
    binWidths.append((10**massBinLimits[i+1]) - (10**massBinLimits[i]))

## ---- Halo mass completeness histogram

count_match, dummy = np.histogram(np.log10(eagle_mass[indexes[:,0]]), 
                                  bins=massBinLimits)

count_full, dummy = np.histogram(np.log10(eagle_mass[unmatched]), 
                                 bins=massBinLimits)

count_total = count_match + count_full


fig,ax1 = plt.subplots(1,1,figsize=(15,4))

ax1.bar(massBins, count_match / count_total, width=binWidths) 
ax1.bar(massBins, count_full / count_total, bottom=count_match / count_total, width=binWidths)

ax1.set_xlabel('Total Subhalo Mass ($M_{\odot}$)', fontsize=13)
ax1.legend(['Matched','Unmatched'], bbox_to_anchor=(1.15, 1), fontsize=13)

ax1.set_xlim(5e7,8e14)
ax1.set_xscale('log')

plt.show()


# ## ---- Stellar mass completeness histogram
# 
# count_match, dummy = np.histogram(np.log10(eagle['MassType_Stars'][indexes[:,0]]), 
#                                   bins=massBinLimits)
# 
# count_full, dummy = np.histogram(np.log10(eagle['MassType_Stars'][unmatched]), 
#                                  bins=massBinLimits)
# 
# count_total = count_match + count_full
# 
# 
# fig,ax1 = plt.subplots(1,1,figsize=(15,4))
# 
# ax1.bar(massBins, count_match / count_total, width=binWidths) 
# ax1.bar(massBins, count_full / count_total, bottom=count_match / count_total, width=binWidths)
# 
# ax1.set_xlabel('Total Subhalo Mass ($M_{\odot}$)', fontsize=13)
# ax1.legend(['Matched','Unmatched'], bbox_to_anchor=(1.15, 1), fontsize=13)
# 
# ax1.set_xlim(5e7,1e12)
# ax1.set_xscale('log')
# 
# plt.show()
# 
