import glob
import h5py
import numpy as np
import pandas as pd

import virgo.formats.subfind_lgadget3 as subfind 


fname = "/cosma5/data/jch/L800/Runs/KEEP/data/groups_272/subhalo_tab_272.*" 
files = glob.glob(fname)

subtab = subfind.SubTabFile(files[0], id_bytes=4) 
subtab.sanity_check()
Ngrp = int(subtab['TotNgroups'][...])
Nsub = int(subtab['TotNsubgroups'][...])

Halo_M_Crit200 = np.zeros(Ngrp)
SubGrNr = np.zeros(Nsub, dtype=int)
SubLen = np.zeros(Nsub, dtype=int)
SubHalfMass = np.zeros(Nsub)
SubVel = np.zeros((Nsub,3))
SubVmax = np.zeros(Nsub)
SubRVmax = np.zeros(Nsub)
SubPotentialEnergy = np.zeros(Nsub)
FirstSub = np.zeros(Ngrp, dtype=int)


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
    SubLen[j:j_max] = subtab["SubLen"][...] 
    SubHalfMass[j:j_max] = subtab["SubHalfMass"][...] 
    SubVel[j:j_max] = subtab["SubVel"][...] 
    SubVmax[j:j_max] = subtab["SubVmax"][...] 
    SubRVmax[j:j_max] = subtab["SubRVmax"][...] 
    SubPotentialEnergy[j:j_max] = subtab["SubPotentialEnergy"][...] 
    FirstSub[i:i_max] = subtab["FirstSub"][...] 

    i += Ngrp_f
    j += Nsub_f


p_mass = 1.06e8 / 0.6777

Satellite = np.ones(Nsub,dtype=int)
Satellite[FirstSub.astype(int)] = 0

data = pd.DataFrame(SubGrNr, columns=['SubGrNr'])
data['FOF_Group_M_Crit200_DM'] = Halo_M_Crit200[SubGrNr.astype(int)] * 1e10
data['M_DM'] = SubLen * p_mass
data['halfMassRad_DM'] = SubHalfMass
data['velocity_DM'] = np.sqrt(np.sum(SubVel**2,axis=1))
data['Vmax_DM'] = SubVmax
data['VmaxRadius_DM'] = SubRVmax
data['PotentialEnergy_DM'] = SubPotentialEnergy
data['Satellite'] = Satellite


output = 'output/'
data.to_csv(output + 'PMillennium' + '_' + 'z000p000' + "_dmo.csv")

