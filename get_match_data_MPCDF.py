# This code was adapted from the python examples on the Eagle wiki given here:
# http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=eagle:documentation:reading_python&s[]=bound


import numpy as np
import math
import sys
import pickle

import eagle as E

from sim_details import mlcosmo
mlc = mlcosmo(ini='config/config_CE-0.ini')

output_folder = 'output/'
nthr = 4

sim = mlc.sim_hydro

numGroups = E.readAttribute("SUBFIND", sim, mlc.tag, "/Header/TotNgroups")
numSubGroups = E.readAttribute("SUBFIND", sim, mlc.tag, "/Header/TotNsubgroups")
boxSize = E.readAttribute("PARTDATA", sim, mlc.tag, "/Header/BoxSize")
hubbleParam = E.readAttribute("PARTDATA", sim, mlc.tag, "/Header/HubbleParam")
H = E.readAttribute("SUBFIND", sim, mlc.tag, "/Header/H(z)") * mlc.Mpc / 1000
rho_crit = 3.*(H / mlc.unitLength)**2  / (8. * math.pi * mlc.G) * mlc.unitMass
rho_bar = E.readAttribute("PARTDATA", sim, mlc.tag, "/Header/Omega0") * rho_crit
redshift = E.readAttribute("PARTDATA", sim, mlc.tag, "/Header/Redshift")
z_int = math.floor(redshift)
z_dec = math.floor(10.*(redshift - z_int))
expansionFactor = E.readAttribute("PARTDATA", sim, mlc.tag, "/Header/ExpansionFactor")
physicalBoxSize = boxSize / hubbleParam

M_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass", numThreads=nthr) * mlc.unitMass
M_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Mass", numThreads=nthr) * mlc.unitMass

lengthType_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)
lengthType_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)


# filter EA and DM arrays before loop
mask_EA = (M_EA > mlc.massLimit) * (lengthType_EA[:,1] > mlc.IDsToMatch)
mask_DM = (M_DM > mlc.massLimit) * (lengthType_DM[:,1] > mlc.IDsToMatch)

del(lengthType_DM,lengthType_EA)


M_EA = M_EA[mask_EA]
M_DM = M_DM[mask_DM]

CoP_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/CentreOfPotential", numThreads=nthr)[mask_EA] * mlc.unitLength
Grp_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, 
                     "Subhalo/GroupNumber", numThreads=nthr)[mask_EA]
Sub_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)[mask_EA]

CoP_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/CentreOfPotential", numThreads=nthr)[mask_DM] * mlc.unitLength
Grp_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)[mask_DM]
Sub_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)[mask_DM]

length_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLength", numThreads=nthr)[mask_EA]
offset_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubOffset", numThreads=nthr)[mask_EA]

length_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLength", numThreads=nthr)[mask_DM]
offset_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubOffset", numThreads=nthr)[mask_DM]

del(mask_DM,mask_EA)

particleIDs_EA = E.readArray("SUBFIND_PARTICLES", mlc.sim_hydro, mlc.tag, "IDs/ParticleID", numThreads=nthr)
particleIDs_DM = E.readArray("SUBFIND_PARTICLES", mlc.sim_dmo, mlc.tag, "IDs/ParticleID", numThreads=nthr)


# # Create rough grid of positions to reduce number of distance calculations
# rat = int(physicalBoxSize * mlc.unitLength / mlc.max_distance)
# d = physicalBoxSize * mlc.unitLength/rat # ~> 8000
# 
# cop_ea_mid = (CoP_EA//d).astype(int)
# 
# cop_ea_up = CoP_EA//d + 1
# cop_ea_up[cop_ea_up > cop_ea_mid.max()] = 0
# 
# cop_ea_down = CoP_EA//d - 1
# cop_ea_down[cop_ea_down < 0] = cop_ea_mid.max()
# 
# cop_dm_mid = CoP_DM//d
# 
# cop_dm_up = CoP_DM//d + 1
# cop_dm_up[cop_dm_up > cop_ea_mid.max()] = 0
# 
# cop_dm_down = CoP_DM//d - 1
# cop_dm_down[cop_dm_down < 0] = cop_ea_mid.max()

# del(CoP_DM,CoP_EA)


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


pickle.dump([M_EA,M_DM,CoP_EA,CoP_DM,particles_DM,bound_particles_EA,
             particles_EA,particles_DM,Grp_EA,Sub_EA,Grp_DM,Sub_DM],
            open(output_folder + mlc.sim_name + "_match_data.p",'wb'))


