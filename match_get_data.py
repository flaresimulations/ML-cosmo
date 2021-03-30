# This code was adapted from the python examples on the Eagle wiki given here:
# http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=eagle:documentation:reading_python&s[]=bound


import numpy as np
import math
import sys
import pickle

import eagle_IO.eagle_IO as E

_config = str(sys.argv[1])

from sim_details import mlcosmo
mlc = mlcosmo(ini=_config)


print("==========\nSim: %s\nTag: %s\n===========\n"%(_config,mlc.tag))

output_folder = 'output/'
nthr = 4


M_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass", numThreads=nthr) * mlc.unitMass
M_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Mass", numThreads=nthr) * mlc.unitMass

lengthType_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)
lengthType_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)


# filter EA and DM arrays before loop
mask_EA = (M_EA > mlc.massLimit) * (lengthType_EA[:,1] > mlc.IDsToMatch)
mask_DM = (M_DM > mlc.massLimit) * (lengthType_DM[:,1] > mlc.IDsToMatch)

del(lengthType_DM,lengthType_EA)

#numHaloes = max(sum(mask_EA),sum(mask_DM))

M_EA = M_EA[mask_EA]
M_DM = M_DM[mask_DM]

CoP_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/CentreOfPotential", numThreads=nthr)[mask_EA] * mlc.unitLength
Grp_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)[mask_EA]
Sub_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)[mask_EA]

CoP_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/CentreOfPotential", numThreads=nthr)[mask_DM] * mlc.unitLength
Grp_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)[mask_DM]
Sub_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)[mask_DM]

length_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLength", numThreads=nthr)[mask_EA]# .astype(long)
offset_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubOffset", numThreads=nthr)[mask_EA]# .astype(long)

length_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLength", numThreads=nthr)[mask_DM]# .astype(long)
offset_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubOffset", numThreads=nthr)[mask_DM]# .astype(long)

del(mask_DM,mask_EA)

particleIDs_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "IDs/ParticleID", numThreads=nthr)
particleIDs_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "IDs/ParticleID", numThreads=nthr)


bound_particles_EA = np.zeros((len(Sub_EA),50),dtype=int)

for i in range(len(Sub_EA)):
    halo_ids = particleIDs_EA[offset_EA[i] : offset_EA[i]+length_EA[i]]
    bound_particles_EA[i,] = halo_ids[halo_ids % 2 == 0][0:mlc.IDsToMatch]


particles_EA = [None] * len(Sub_EA)
particles_DM = [None] * len(Sub_DM)

for i in range(len(Sub_EA)):
    particles_EA[i] = particleIDs_EA[offset_EA[i] : offset_EA[i]+length_EA[i]]

for i in range(len(Sub_DM)):
    particles_DM[i] = particleIDs_DM[offset_DM[i] : offset_DM[i]+length_DM[i]]

del(particleIDs_EA,particleIDs_DM)
del(offset_EA,offset_DM,length_EA,length_DM)


pickle.dump([M_EA,M_DM,CoP_EA,CoP_DM,bound_particles_EA,
             particles_EA,particles_DM,Grp_EA,Sub_EA,Grp_DM,Sub_DM],
            open(output_folder + mlc.sim_name + '_' + mlc.tag + "_match_data.p",'wb'))

# pickle.dump([M_EA,CoP_EA,bound_particles_EA,particles_EA,Grp_EA,Sub_EA],
#             open(output_folder + mlc.sim_name + "_match_data_EA.p",'wb'))
# 
# pickle.dump([M_DM,CoP_DM,particles_DM,Grp_DM,Sub_DM],
#             open(output_folder + mlc.sim_name + "_match_data_DM.p",'wb'))


