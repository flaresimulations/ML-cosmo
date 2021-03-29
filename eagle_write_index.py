"""
Read match file, load reference and dark matter only sims, match properties and write out
"""
import sys

import numpy as np
import pandas as pd

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo
_config = str(sys.argv[1])
mlc = mlcosmo(ini=_config)

density = True
nthr=16

output = 'output/'

f = "%s/matchedHalosSub_%s_%s.dat"%(output,mlc.sim_name,mlc.tag)
match = pd.read_csv(f)
match.reset_index(inplace=True)


Sub_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)
Grp_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)

Grp_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)
Sub_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)


# ---- Find index of matches in subhalo arrays
_fname = output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt'
    
idx_EA = []
idx_DM = []
for i in range(len(match)):
    idx_EA.append( np.where((Grp_EA == match['Grp_EA'][i]) & (Sub_EA == match['Sub_EA'][i]))[0][0] )
    idx_DM.append( np.where((Grp_DM == match['Grp_DM'][i]) & (Sub_DM == match['Sub_DM'][i]))[0][0] )

np.savetxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt', 
           np.array([idx_EA,idx_DM]).T, fmt='%i')

