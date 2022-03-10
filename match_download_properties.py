"""
Read match file, load reference and dark matter only sims, match properties and write out
"""
import os, sys, math

import glob
import numpy as np
import pandas as pd

import eagle_IO.eagle_IO as E

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file", type=str)
parser.add_argument("--region", help="if a flares zoom region, provide the region number",
                    type=int, default=None)
parser.add_argument("--tag", help="snapshot tag string", type=str, default=None)
parser.add_argument("--density", help="download local density features", action='store_true')

args = parser.parse_args()

from sim_details import mlcosmo
mlc = mlcosmo(ini=args.config, region=args.region, tag=args.tag)


# from sim_details import mlcosmo
# _config = str(sys.argv[1])
# mlc = mlcosmo(ini=_config)

nthr=16
output = 'output/'

f = "%s/matchedHalosSub_%s_%s.dat"%(output,mlc.sim_name,mlc.tag)
match = pd.read_csv(f)
match.reset_index(inplace=True)


Sub_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)
Grp_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)

Grp_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)
Sub_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)


_idx = np.loadtxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt')

idx_EA = _idx[:,0].astype(int)
idx_DM = _idx[:,1].astype(int)  

# ---- Initialise Dataframe

data = pd.DataFrame(idx_EA,columns=['idx_EA'])
data['idx_DM'] = idx_DM

data['Grp_EA'] = Grp_EA[idx_EA]
data['Grp_DM'] = Grp_DM[idx_DM]
data['Sub_EA'] = Sub_EA[idx_EA]
data['Sub_DM'] = Sub_DM[idx_DM]

# ---- Read Dark matter properties

data['halfMassProjRad_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/HalfMassProjRad", numThreads=nthr, noH=True)[idx_DM,1] * mlc.unitLength
data['halfMassRad_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/HalfMassRad", numThreads=nthr, noH=True)[idx_DM,1] * mlc.unitLength

data['KE_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/KineticEnergy", numThreads=nthr, noH=True)[idx_DM]
data['TE_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/TotalEnergy", numThreads=nthr, noH=True)[idx_DM]
data['PotentialEnergy_DM'] = data['TE_DM'] - data['KE_DM']


data['M_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Mass", noH=True)[idx_DM] * mlc.unitMass
data['MassTwiceHalfMassRad_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/MassTwiceHalfMassRad", numThreads=nthr, noH=True)[idx_DM,1] * mlc.unitMass

_vel = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Velocity", numThreads=nthr, noH=True)[idx_DM,:]
data['velocity_DM'] = np.sqrt(np.sum(_vel**2, axis=1))

data['Vmax_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Vmax", numThreads=nthr, noH=True)[idx_DM]
data['VmaxRadius_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/VmaxRadius", numThreads=nthr, noH=True)[idx_DM]

data['length_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLength", numThreads=nthr, noH=True)[idx_DM]


## Match FOF properties by subhalo group number

data['FOF_NumOfSubhalos_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/NumOfSubhalos", numThreads=nthr, noH=True)[data['Grp_DM']]

data['FOF_GroupMass_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/GroupMass", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
data['FOF_GroupLength_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/GroupLength", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength
# data['FOF_ContaminationMass_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/ContaminationMass", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
data['FOF_Group_M_Crit200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Crit200", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
# data['FOF_Group_M_Crit2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Crit2500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
# data['FOF_Group_M_Crit500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Crit500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
# data['FOF_Group_M_Mean200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Mean200", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
# data['FOF_Group_M_Mean2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Mean2500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
# data['FOF_Group_M_Mean500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Mean500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass
# data['FOF_Group_M_TopHat200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_TopHat200", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitMass

data['FOF_Group_R_Crit200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Crit200", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength
# data['FOF_Group_R_Crit2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Crit2500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength
# data['FOF_Group_R_Crit500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Crit500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength
# data['FOF_Group_R_Mean200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Mean200", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength
# data['FOF_Group_R_Mean2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Mean2500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength
# data['FOF_Group_R_Mean500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Mean500", numThreads=nthr, noH=True)[data['Grp_DM']-1] * mlc.unitLength



# ---- Read Full Eagle properties

data['M_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass", numThreads=nthr, noH=True)[idx_EA] * mlc.unitMass
# data['length_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLength", numThreads=nthr, noH=True)[idx_EA]
# 
# lengthType_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)[idx_EA]
# data['lengthType_Gas_EA'] = lengthType_EA[:,0]
# data['lengthType_Stars_EA'] = lengthType_EA[:,4]
# data['lengthType_BH_EA'] = lengthType_EA[:,5]

data['BlackHoleMass_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMass", numThreads=nthr, noH=True)[idx_EA] * mlc.unitMass
data['BlackHoleMassAccretionRate_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMassAccretionRate", numThreads=nthr, noH=True)[idx_EA]

GasSpin_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GasSpin", numThreads=nthr, noH=True)[idx_EA]
data['GasSpin_EA'] = pow(GasSpin_EA[:,0]**2 + GasSpin_EA[:,1]**2 + GasSpin_EA[:,2]**2,0.5)

halfMassProjRad_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassProjRad", numThreads=nthr, noH=True)[idx_EA] * mlc.unitLength
data['halfMassProjRad_Gas_EA'] = halfMassProjRad_EA[:,0]
data['halfMassProjRad_Stars_EA'] = halfMassProjRad_EA[:,4]
data['halfMassProjRad_BH_EA'] = halfMassProjRad_EA[:,5]

halfMassRad_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassRad", numThreads=nthr, noH=True)[idx_EA] * mlc.unitLength
data['halfMassRad_Gas_EA'] = halfMassRad_EA[:,0]
data['halfMassRad_Stars_EA'] = halfMassRad_EA[:,4]
data['halfMassRad_BH_EA'] = halfMassRad_EA[:,5]


data['KineticEnergy_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/KineticEnergy", numThreads=nthr, noH=True)[idx_EA]

# data['InitialMassWeightedBirthZ_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/InitialMassWeightedBirthZ", numThreads=nthr)[idx_EA]
# data['InitialMassWeightedStellarAge_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/InitialMassWeightedStellarAge", numThreads=nthr)[idx_EA]

# MassType_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassType", numThreads=nthr, noH=True)[idx_EA] * mlc.unitMass
# data['MassType_Gas_EA'] = MassType_EA[:,0]
# data['MassType_Stars_EA'] = MassType_EA[:,4]
# data['MassType_BH_EA'] = MassType_EA[:,5]

MassTwiceHalfMassRad_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassTwiceHalfMassRad", numThreads=nthr, noH=True)[idx_EA] * mlc.unitMass
data['MassTwiceHalfMassRad_Gas_EA'] = MassTwiceHalfMassRad_EA[:,0]
data['MassTwiceHalfMassRad_Stars_EA'] = MassTwiceHalfMassRad_EA[:,4]
data['MassTwiceHalfMassRad_BH_EA'] = MassTwiceHalfMassRad_EA[:,5]


# data['StellarInitialMass_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarInitialMass", numThreads=nthr)[idx_EA] * mlc.unitMass
# data['Stars_Mass_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Mass", numThreads=nthr, noH=True)[idx_EA] * mlc.unitMass

aperture_mass = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/ApertureMeasurements/Mass/030kpc", numThreads=nthr, noH=True)
data['Stars_Mass_EA'] = aperture_mass[idx_EA,4] * mlc.unitMass
data['MassType_Gas_EA'] = aperture_mass[idx_EA,0] * mlc.unitMass

Stars_Spin_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Spin", numThreads=nthr, noH=True)[idx_EA]
data['Stars_Spin_EA'] = pow(Stars_Spin_EA[:,0]**2 + Stars_Spin_EA[:,1]**2 + Stars_Spin_EA[:,2]**2,0.5)

data['Stars_Metallicity_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Metallicity", numThreads=nthr, noH=True)[idx_EA]

data['StellarVelDisp_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarVelDisp", numThreads=nthr, noH=True)[idx_EA]
data['StellarVelDisp_HalfMassProjRad_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarVelDisp_HalfMassProjRad", numThreads=nthr, noH=True)[idx_EA]

data['StarFormationRate_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StarFormationRate", numThreads=nthr, noH=True)[idx_EA]

# data['Subhalo_Mass_DM'] = 1.15*10**7 * data['length_DM']
data['Satellite'] = (data['Sub_DM'] != 0).astype(int)

if args.density:
    data['Density_R1'] = np.loadtxt('output/density_' + mlc.sim_name + '_' + mlc.tag + '_R1.txt')
    data['Density_R2'] = np.loadtxt('output/density_' + mlc.sim_name + '_' + mlc.tag + '_R2.txt')
    data['Density_R4'] = np.loadtxt('output/density_' + mlc.sim_name + '_' + mlc.tag + '_R4.txt')
    data['Density_R8'] = np.loadtxt('output/density_' + mlc.sim_name + '_' + mlc.tag + '_R8.txt')
    # data['Density_R16'] = np.loadtxt('output/density_' + mlc.sim_name + '_' + mlc.tag + '_R16.txt')


data.to_csv(output + mlc.sim_name + '_' + mlc.tag + "_match.csv")

