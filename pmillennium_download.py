import random

import glob
import h5py
import numpy as np
import pandas as pd

import virgo.formats.subfind_lgadget3 as subfind 

## [Constants]
G = 4.302e4
unitMass = 1e10
unitLength = 1e3
Mpc = 3.08567758e22
_h = 0.6777

snap = '249' # z = 0.1099
redshift = 'z000p101'
fname = f"/cosma5/data/jch/L800/Runs/KEEP/data/groups_{snap}/subhalo_tab_{snap}.*" 
files = glob.glob(fname)

subtab = subfind.SubTabFile(files[0], id_bytes=4) 
subtab.sanity_check()
Ngrp = int(subtab['TotNgroups'][...])
Nsub = int(subtab['TotNsubgroups'][...])

Halo_M_Crit200 = np.zeros(Ngrp)
SubGrNr = np.zeros(Nsub, dtype=int)
GrNr = np.zeros(Ngrp, dtype=int)
SubLen = np.zeros(Nsub, dtype=int)
SubHalfMass = np.zeros(Nsub)
SubVel = np.zeros((Nsub,3))
SubPos = np.zeros((Nsub,3))
SubVmax = np.zeros(Nsub)
SubRVmax = np.zeros(Nsub)
Nsubs = np.zeros(Ngrp, dtype=int)
SubPotentialEnergy = np.zeros(Nsub)
SubBindingEnergy = np.zeros(Nsub)
Satellite = np.ones(Nsub,dtype=int)


i,j = 0,0
for f in files:
    print(f)
    subtab = subfind.SubTabFile(f, id_bytes=4) 
    subtab.sanity_check()

    Ngrp_f = int(subtab['Ngroups'][...])
    Nsub_f = int(subtab['Nsubgroups'][...])

    i_max = i + Ngrp_f
    j_max = j + Nsub_f

    Halo_M_Crit200[i:i_max] = subtab["Halo_M_Crit200"][...] 
    SubGrNr[j:j_max] = subtab["SubGrNr"][...] 
    GrNr[i:i_max] = subtab["GroupNr"][...] 
    SubLen[j:j_max] = subtab["SubLen"][...] 
    SubHalfMass[j:j_max] = subtab["SubHalfMass"][...] 
    SubVel[j:j_max] = subtab["SubVel"][...] 
    SubPos[j:j_max] = subtab["SubPos"][...] 
    SubVmax[j:j_max] = subtab["SubVmax"][...] 
    SubRVmax[j:j_max] = subtab["SubRVmax"][...] 
    SubPotentialEnergy[j:j_max] = subtab["SubPotentialEnergy"][...]
    SubBindingEnergy[j:j_max] = subtab["SubBindingEnergy"][...]
    Nsubs[i:i_max] = subtab["Nsubs"][...]
    FirstSub = subtab["FirstSub"][...] 
    Satellite[j:j_max][FirstSub[Nsubs[i:i_max] > 0]] = 0

    i += Ngrp_f
    j += Nsub_f


data = pd.DataFrame(SubGrNr, columns=['SubGrNr'])

Gr_indexes = np.zeros(len(GrNr),dtype=int)
Gr_indexes[GrNr] = np.arange(len(GrNr))
data['FOF_Group_M_Crit200_DM'] = (Halo_M_Crit200[Gr_indexes[SubGrNr]]*1e10)/_h
data['Nsubs'] = Nsubs[Gr_indexes[SubGrNr]]

p_mass = 1.06e8 / _h
data['M_DM'] = SubLen * p_mass
data['halfMassRad_DM'] = (SubHalfMass * unitLength) / _h
data['velocity_DM'] = np.sqrt(np.sum(SubVel**2,axis=1))
data['Vmax_DM'] = SubVmax
data['VmaxRadius_DM'] = SubRVmax / _h
data['PotentialEnergy_DM'] = SubPotentialEnergy * (1./unitLength) 
# data['BindingEnergy_DM'] = SubBindingEnergy
data['Satellite'] = Satellite
data['SubPos_x'] = SubPos[:,0] / _h
data['SubPos_y'] = SubPos[:,1] / _h
data['SubPos_z'] = SubPos[:,2] / _h

output = 'output/'
data.to_csv(output + 'PMillennium_' + redshift + "_dmo.csv")

# filename = output + 'PMillennium' + '_' + 'z000p000' + "_dmo.csv"
mask = (SubPos[:,0] < 100) & (SubPos[:,1] < 100) & (SubPos[:,2] < 100)
df = data[mask]

# n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
# skip = np.where(np.random.rand(n) > 0.05)[0]
# df = pd.read_csv(filename, skiprows=skip)

df.to_csv(output + 'PMillennium_' + redshift + "_dmo_subset.csv")

