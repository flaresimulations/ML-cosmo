# This code was adapted from the python examples on the Eagle wiki given here:
# http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=eagle:documentation:reading_python&s[]=bound

import os
import numpy as np
import math
import sys
import h5py

import eagle_IO.eagle_IO as E

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file", type=str)
parser.add_argument("--region", help="if a flares zoom region, provide the region number", 
                    type=int, default=None)
parser.add_argument("--tag", help="snapshot tag string", type=str, default=None)
args = parser.parse_args()

from sim_details import mlcosmo
mlc = mlcosmo(ini=args.config, region=args.region, tag=args.tag)


print("==========\nSim: %s\nTag: %s\n===========\n"%(mlc.sim_name,mlc.tag))

output_folder = 'output/%s/'%mlc.tag
if not os.path.isdir(output_folder):
    print("Creating folder:", output_folder)
    os.mkdir(output_folder)

nthr = 8


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


with h5py.File(output_folder + mlc.sim_name + '_' + mlc.tag + "_match_data.h5", 'w') as hf:
    hf.create_dataset('M_EA', data=M_EA)
    hf.create_dataset('M_DM', data=M_DM)
    hf.create_dataset('CoP_EA', data=CoP_EA)
    hf.create_dataset('CoP_DM', data=CoP_DM)
    hf.create_dataset('bound_particles_EA', data=bound_particles_EA)
    hf.create_dataset('Grp_EA', data=Grp_EA)
    hf.create_dataset('Grp_DM', data=Grp_DM)
    hf.create_dataset('Sub_EA', data=Sub_EA)
    hf.create_dataset('Sub_DM', data=Sub_DM)

    dt = h5py.vlen_dtype(np.dtype('int64'))  # variable length for ragged arrays
    hf.create_dataset('particles_EA', data=particles_EA, dtype=dt)
    hf.create_dataset('particles_DM', data=particles_DM, dtype=dt)


