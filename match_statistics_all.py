import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo


output = 'output/'


def count_plots(configs):
    massBinLimits = np.linspace(5.2, 15.4, 52)
    massBins = np.logspace(5.3, 15.3, 51)
    binWidths = []
    for i,z in enumerate(massBins):
        binWidths.append((10**massBinLimits[i+1]) - (10**massBinLimits[i]))

    m_limit = 1e9
    count_plot = {}
    
    for _config in configs:
        mlc = mlcosmo(ini=_config)

        eagle_match = pd.read_csv(output + mlc.sim_name + '_' + mlc.tag + '_match.csv')
        indexes = np.loadtxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt', dtype=int)

        eagle_mass = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Mass") * 1e10

        unmatched = ~np.in1d(range(len(eagle_mass)), indexes[:,1]) 
        # print(_config, '\n', np.sum(eagle_mass[indexes[:,1]] > m_limit) / np.sum(eagle_mass > m_limit))

        count_match, dummy = np.histogram(np.log10(eagle_mass[indexes[:,1]]), 
                                          bins=massBinLimits)

        count_full, dummy = np.histogram(np.log10(eagle_mass[unmatched]), 
                                         bins=massBinLimits)

        count_total = count_match + count_full
        count_plot[_config] = count_match / count_total

    return count_plot


## C-EAGLE match stats (run on mpcdf)
configs = ['config/config_CE-%i.ini'%i for i in np.arange(30)]
count_plot = count_plots(configs)
count_plot = {key: list(count_plot[key]) for key,item in count_plot.items()}
with open('output/match_stats_mpcdf.json','w') as f: 
    json.dump(count_plot,f)


## Periodic match stats
configs = [#'config/config_cosma_L0050N0752.ini',
           #'config/config_cosma_L0100N1504.ini',
           'config/config_CE-0.ini',
           'config/config_CE-1.ini']

configs_pretty = ['L0050N0752','L0100N1504']

count_plot = count_plots(configs)


fig,ax1 = plt.subplots(1,1,figsize=(15,4))

for _config,pretty in zip(configs,configs_pretty):
    ax1.step(massBins, count_plot[_config], label=pretty, lw=3) 

ax1.set_xlabel('$M_{\mathrm{subhalo}} \,/\, \mathrm{M_{\odot}}$', fontsize=13)
ax1.set_ylabel('$f_{\mathrm{match}}$', fontsize=13)
ax1.legend(loc=4)#bbox_to_anchor=(1.15, 1), fontsize=13)

ax1.set_xlim(1e8,8e14)
ax1.hlines(1.0,ax1.get_xlim()[0],ax1.get_xlim()[1], linestyle='dashed', alpha=0.5)
ax1.set_ylim(0,1.05)
ax1.set_xscale('log')

plt.show()
# fname = 'plots/match_statistics.png'
# plt.savefig(fname, dpi=150, bbox_inches='tight')
