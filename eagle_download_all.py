##
## Read eagle attributes and arrays on cosma
## write out to pickle files for local analysis
##

import pickle as pcl
import eagle as E
import sys


import glob
import os
#os.chdir('/home/chris/sussex/cosmo-sim-ML/scripts/')

import pandas as pd

from sim_details import mlcosmo
mlc = mlcosmo()

output = 'output/'



# ---- Read Full Eagle properties

Sub = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber")
Grp = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GroupNumber")

# initialise dataframe
data = pd.DataFrame(Sub,columns=['Sub'])
data['Grp'] = Grp

# data['halfMassProjRad'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassProjRad")[:,1] * mlc.unitLength
# data['halfMassRad'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassRad")[:,1] * mlc.unitLength

# data['KE'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/KineticEnergy")
# data['TE'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/TotalEnergy")

# data['MassTwiceHalfMassRad'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassTwiceHalfMassRad")[:,1] * mlc.unitMass
# 
# data['velocity'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Velocity")[:,1]
# data['Vmax'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Vmax")
# data['VmaxRadius'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/VmaxRadius")
# 
# data['length'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLength")
# 
# data['FOF_NumOfSubhalos'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/NumOfSubhalos")
# 
# data['FOF_GroupMass'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/GroupMass") * mlc.unitMass
# data['FOF_GroupLength'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/GroupLength") * mlc.unitLength
# data['FOF_ContaminationMass'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/ContaminationMass") * mlc.unitMass
# data['FOF_Group_M_Crit200'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_Crit200") * mlc.unitMass
# data['FOF_Group_M_Crit2500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_Crit2500") * mlc.unitMass
# data['FOF_Group_M_Crit500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_Crit500") * mlc.unitMass
# data['FOF_Group_M_Mean200'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_Mean200") * mlc.unitMass
# data['FOF_Group_M_Mean2500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_Mean2500") * mlc.unitMass
# data['FOF_Group_M_Mean500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_Mean500") * mlc.unitMass
# data['FOF_Group_M_TopHat200'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_M_TopHat200") * mlc.unitMass
# 
# data['FOF_Group_R_Crit200'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_R_Crit200") * mlc.unitLength
# data['FOF_Group_R_Crit2500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_R_Crit2500") * mlc.unitLength
# data['FOF_Group_R_Crit500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_R_Crit500") * mlc.unitLength
# data['FOF_Group_R_Mean200'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_R_Mean200") * mlc.unitLength
# data['FOF_Group_R_Mean2500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_R_Mean2500") * mlc.unitLength
# data['FOF_Group_R_Mean500'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "FOF/Group_R_Mean500") * mlc.unitLength



data['Subhalo_Mass'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass") * mlc.unitMass

# lengthType = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLengthType")
# data['lengthType_Gas'] = lengthType[:,0]
# data['lengthType_Stars'] = lengthType[:,4]
# data['lengthType_BH'] = lengthType[:,5]
# 
# data['BlackHoleMass'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMass")
# data['BlackHoleMassAccretionRate'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/BlackHoleMassAccretionRate")
# 
# 
# data['GasSpin_x'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GasSpin")[:,0]
# data['GasSpin_y'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GasSpin")[:,1]
# data['GasSpin_z'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GasSpin")[:,2]
# #data['GasSpin'] = pow(GasSpin_EA[:,0]**2 + GasSpin_EA[:,1]**2 + GasSpin_EA[:,2]**2,0.5)
# 
# halfMassProjRad = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassProjRad") * mlc.unitLength
# data['halfMassProjRad_Gas'] = halfMassProjRad[:,0]
# data['halfMassProjRad_Stars'] = halfMassProjRad[:,4]
# data['halfMassProjRad_BH'] = halfMassProjRad[:,5]
# 
# halfMassRad = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/HalfMassRad") * mlc.unitLength
# data['halfMassRad_Gas'] = halfMassRad[:,0]
# data['halfMassRad_Stars'] = halfMassRad[:,4]
# data['halfMassRad_BH'] = halfMassRad[:,5]
# 
# 
# # InertiaTensor_EA = E.readArray("SUBFIND", sim_EA, tag, "Subhalo/InertiaTensor") # TODO: Throws a read error, investigate
# data['KineticEnergy'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/KineticEnergy")
# 
# data['InitialMassWeightedBirthZ'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/InitialMassWeightedBirthZ")
# data['InitialMassWeightedStellarAge'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/InitialMassWeightedStellarAge")

MassType = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassType") * mlc.unitMass
data['MassType_DarkMatter'] = MassType[:,1]
data['MassType_Gas'] = MassType[:,0]
data['MassType_Stars'] = MassType[:,4]
data['MassType_BH'] = MassType[:,5]

# MassTwiceHalfMassRad = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/MassTwiceHalfMassRad") * mlc.unitMass
# data['MassTwiceHalfMassRad_Gas'] = MassTwiceHalfMassRad[:,0]
# data['MassTwiceHalfMassRad_Stars'] = MassTwiceHalfMassRad[:,4]
# data['MassTwiceHalfMassRad_BH'] = MassTwiceHalfMassRad[:,5]
# 
# 
# data['StellarInitialMass'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarInitialMass") * mlc.unitMass
# data['Stars_Mass'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Mass") * mlc.unitMass
# 
# 
# data['Stars_Spin_x'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Spin")[:,0]
# data['Stars_Spin_y'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Spin")[:,1]
# data['Stars_Spin_z'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Spin")[:,2]
# #data['Stars_Spin'] = pow(Stars_Spin_EA[:,0]**2 + Stars_Spin_EA[:,1]**2 + Stars_Spin_EA[:,2]**2,0.5)
# 
# data['Stars_Metallicity'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Stars/Metallicity")
# 
# data['StellarVelDisp'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarVelDisp")
# data['StellarVelDisp_HalfMassProjRad'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StellarVelDisp_HalfMassProjRad")
# 
# data['StarFormationRate'] = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/StarFormationRate")



data.to_csv(output + mlc.tag + "_all.csv")

# pcl.dump(master, open(output + tag + ".p", "wb"))


