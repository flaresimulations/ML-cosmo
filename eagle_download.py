"""
Read match file, load reference and dark matter only sims, match properties and write out
"""
import sys
import eagle_IO.eagle_IO as E
import math
import numpy as np

import glob

import pandas as pd

_config = str(sys.argv[1])


from sim_details import mlcosmo
mlc = mlcosmo(ini=_config)

redshift = float(mlc.tag[5:].replace('p','.')) 
z_int = math.floor(redshift)
z_dec = math.floor(10.*(redshift - z_int))


output = 'output/'

f = "%s/matchedHalosSub_%s_%s.dat"%(output,mlc.sim_name,mlc.tag)
match = pd.read_csv(f)
match.reset_index(inplace=True)


Sub_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber")
Grp_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GroupNumber")

Grp_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/GroupNumber")
Sub_DM = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubGroupNumber")


# ---- Find index of matches in subhalo arrays

idx_EA = []
idx_DM = []
for i in range(len(match)):
    idx_EA.append( np.where((Grp_EA == match['Grp_EA'][i]) & (Sub_EA == match['Sub_EA'][i]))[0][0] )
    idx_DM.append( np.where((Grp_DM == match['Grp_DM'][i]) & (Sub_DM == match['Sub_DM'][i]))[0][0] )

np.savetxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt', np.array([idx_EA,idx_DM]).T, fmt='%i')


# ---- Initialise Dataframe

data = pd.DataFrame(idx_EA,columns=['idx_EA'])
data['idx_DM'] = idx_DM

data['Grp_EA'] = Grp_EA[idx_EA]
data['Grp_DM'] = Grp_DM[idx_DM]
data['Sub_EA'] = Sub_EA[idx_EA]
data['Sub_DM'] = Sub_DM[idx_DM]

# ---- Read Dark matter properties

data['halfMassProjRad_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/HalfMassProjRad")[idx_DM,1] * mlc.unitLength
data['halfMassRad_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/HalfMassRad")[idx_DM,1] * mlc.unitLength

data['KE_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/KineticEnergy")[idx_DM]
data['TE_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/TotalEnergy")[idx_DM]  #TODO: what's included in this?

#data['M_DM'] = E.read_array("SUBFIND", sim_DM, tag, "Subhalo/MassType_DM")[idx_DM] * unitMass doesn't exist for DMO, need to multiply particle number by dark matter mass
data['MassTwiceHalfMassRad_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/MassTwiceHalfMassRad")[idx_DM,1] * mlc.unitMass

data['velocity_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Velocity")[idx_DM,1]
data['Vmax_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Vmax")[idx_DM]
data['VmaxRadius_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/VmaxRadius")[idx_DM]

data['length_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLength")[idx_DM]


## Match FOF properties by subhalo group number

data['FOF_NumOfSubhalos_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/NumOfSubhalos")[data['Grp_DM']]

data['FOF_GroupMass_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/GroupMass")[data['Grp_DM']] * mlc.unitMass
data['FOF_GroupLength_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/GroupLength")[data['Grp_DM']] * mlc.unitLength
data['FOF_ContaminationMass_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/ContaminationMass")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_Crit200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Crit200")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_Crit2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Crit2500")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_Crit500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Crit500")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_Mean200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Mean200")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_Mean2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Mean2500")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_Mean500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_Mean500")[data['Grp_DM']] * mlc.unitMass
data['FOF_Group_M_TopHat200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_M_TopHat200")[data['Grp_DM']] * mlc.unitMass

data['FOF_Group_R_Crit200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Crit200")[data['Grp_DM']] * mlc.unitLength
data['FOF_Group_R_Crit2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Crit2500")[data['Grp_DM']] * mlc.unitLength
data['FOF_Group_R_Crit500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Crit500")[data['Grp_DM']] * mlc.unitLength
data['FOF_Group_R_Mean200_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Mean200")[data['Grp_DM']] * mlc.unitLength
data['FOF_Group_R_Mean2500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Mean2500")[data['Grp_DM']] * mlc.unitLength
data['FOF_Group_R_Mean500_DM'] = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, "FOF/Group_R_Mean500")[data['Grp_DM']] * mlc.unitLength



# ---- Read Full Eagle properties

data['M_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass")[idx_EA] * mlc.unitMass

data['length_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLength")[idx_EA]

lengthType_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLengthType")[idx_EA]
data['lengthType_Gas_EA'] = lengthType_EA[:,0]
data['lengthType_Stars_EA'] = lengthType_EA[:,4]
data['lengthType_BH_EA'] = lengthType_EA[:,5]

data['BlackHoleMass_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMass")[idx_EA]
data['BlackHoleMassAccretionRate_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMassAccretionRate")[idx_EA]

GasSpin_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GasSpin")[idx_EA]
data['GasSpin_EA'] = pow(GasSpin_EA[:,0]**2 + GasSpin_EA[:,1]**2 + GasSpin_EA[:,2]**2,0.5)

halfMassProjRad_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassProjRad")[idx_EA] * mlc.unitLength
data['halfMassProjRad_Gas_EA'] = halfMassProjRad_EA[:,0]
data['halfMassProjRad_Stars_EA'] = halfMassProjRad_EA[:,4]
data['halfMassProjRad_BH_EA'] = halfMassProjRad_EA[:,5]

halfMassRad_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassRad")[idx_EA] * mlc.unitLength
data['halfMassRad_Gas_EA'] = halfMassRad_EA[:,0]
data['halfMassRad_Stars_EA'] = halfMassRad_EA[:,4]
data['halfMassRad_BH_EA'] = halfMassRad_EA[:,5]


data['KineticEnergy_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/KineticEnergy")[idx_EA]

data['InitialMassWeightedBirthZ_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/InitialMassWeightedBirthZ")[idx_EA]
data['InitialMassWeightedStellarAge_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/InitialMassWeightedStellarAge")[idx_EA]

MassType_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassType")[idx_EA] * mlc.unitMass
data['MassType_Gas_EA'] = MassType_EA[:,0]
data['MassType_Stars_EA'] = MassType_EA[:,4]
data['MassType_BH_EA'] = MassType_EA[:,5]

MassTwiceHalfMassRad_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassTwiceHalfMassRad")[idx_EA] * mlc.unitMass
data['MassTwiceHalfMassRad_Gas_EA'] = MassTwiceHalfMassRad_EA[:,0]
data['MassTwiceHalfMassRad_Stars_EA'] = MassTwiceHalfMassRad_EA[:,4]
data['MassTwiceHalfMassRad_BH_EA'] = MassTwiceHalfMassRad_EA[:,5]


data['StellarInitialMass_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarInitialMass")[idx_EA] * mlc.unitMass
data['Stars_Mass_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Mass")[idx_EA] * mlc.unitMass

Stars_Spin_EA = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Spin")[idx_EA]
data['Stars_Spin_EA'] = pow(Stars_Spin_EA[:,0]**2 + Stars_Spin_EA[:,1]**2 + Stars_Spin_EA[:,2]**2,0.5)

data['Stars_Metallicity_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Metallicity")[idx_EA]

data['StellarVelDisp_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarVelDisp")[idx_EA]
data['StellarVelDisp_HalfMassProjRad_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarVelDisp_HalfMassProjRad")[idx_EA]

data['StarFormationRate_EA'] = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StarFormationRate")[idx_EA]


_df = pd.DataFrame(data)
_df.to_csv(output + mlc.sim_name + '_' + mlc.tag + "_match.csv")


