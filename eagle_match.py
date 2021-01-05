# This code was adapted from the python examples on the Eagle wiki given here:
# http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=eagle:documentation:reading_python&s[]=bound


import numpy as np
import math
import sys

import eagle as E

from sim_details import mlcosmo
mlc = mlcosmo()

# sim_tag = '/virgo/simulations/Hydrangea/C-EAGLE/HaloF0/'
# sim_name = 'L0012N0376'
# sim_tag = '/virgo/simulations/Eagle/L0012N0376/PE/'
# sim_EA = sim_tag+'REFERENCE/data'
# sim_DM = sim_tag+'DMONLY/data'
# tag = '028_z000p000'
# fracToFind = 0.5
# IDsToMatch = 50
# massLimit = 5e8
# N_min = 5000
# mass_diff = 6       # mass ratio
# max_distance = 2000 # kpc
# G = 4.302e4; # kpc (1e10 Mo)^-1 km^2 s^-2
# unitMass = 1e10
# unitLength = 1e3
# Mpc = 3.08567758e22

output_folder = 'output/'

nthr = 4

## serial arguments
# rank = 0
# jobs = 1
rank = int(sys.argv[1])  # rank of process
jobs = int(sys.argv[2])  # total number of processes

# for sim in sims:
sim = mlc.sim_hydro
# General information
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
# softeningComoving = E.readAttribute("PARTDATA", sim, mlc.tag, "/RuntimePars/SofteningGas") * unitLength
# softeningMaxPhys = E.readAttribute("PARTDATA", sim, mlc.tag, "/RuntimePars/SofteningGasMaxPhys") * unitLength
# softening = min(softeningMaxPhys, softeningComoving * expansionFactor)



M_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/Mass", numThreads=nthr) * mlc.unitMass
M_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/Mass", numThreads=nthr) * mlc.unitMass

lengthType_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)
lengthType_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLengthType", numThreads=nthr)


# filter EA and DM arrays before loop
mask_EA = (M_EA > mlc.massLimit) * (lengthType_EA[:,1] > mlc.IDsToMatch)
mask_DM = (M_DM > mlc.massLimit) * (lengthType_DM[:,1] > mlc.IDsToMatch)

del(lengthType_DM,lengthType_EA)

#numHaloes = max(sum(mask_EA),sum(mask_DM))

M_EA = M_EA[mask_EA]
M_DM = M_DM[mask_DM]

CoP_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/CentreOfPotential", numThreads=nthr)[mask_EA] * mlc.unitLength
Grp_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)[mask_EA]
Sub_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)[mask_EA]

CoP_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/CentreOfPotential", numThreads=nthr)[mask_DM] * mlc.unitLength
Grp_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/GroupNumber", numThreads=nthr)[mask_DM]
Sub_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubGroupNumber", numThreads=nthr)[mask_DM]

length_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubLength", numThreads=nthr)[mask_EA]# .astype(long)
offset_EA = E.readArray("SUBFIND", mlc.sim_hydro, mlc.tag, "Subhalo/SubOffset", numThreads=nthr)[mask_EA]# .astype(long)

length_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubLength", numThreads=nthr)[mask_DM]# .astype(long)
offset_DM = E.readArray("SUBFIND", mlc.sim_dmo, mlc.tag, "Subhalo/SubOffset", numThreads=nthr)[mask_DM]# .astype(long)

del(mask_DM,mask_EA)

particleIDs_EA = E.readArray("SUBFIND_PARTICLES", mlc.sim_hydro, mlc.tag, "IDs/ParticleID", numThreads=nthr) #oldSubfindFix=True)#, numThreads=16)
particleIDs_DM = E.readArray("SUBFIND_PARTICLES", mlc.sim_dmo, mlc.tag, "IDs/ParticleID", numThreads=nthr) #oldSubfindFix=True)#, numThreads=16)

match_ea = np.array([])#,dtype=uint64)
match_dm = np.array([])#,dtype=uint64)

# matcher(particleIDs_EA, particleIDs_DM, length_EA, length_DM, offset_EA, offset_DM, Grp_EA, Grp_DM, Sub_EA, Sub_DM, M_EA, M_DM, len(M_EA), len(M_DM), mass_diff, fracToFind, IDsToMatch)


# Create rough grid of positions to reduce number of distance calculations
rat = int(physicalBoxSize * mlc.unitLength / mlc.max_distance)
d = physicalBoxSize * mlc.unitLength/rat # ~> 8000

cop_ea_mid = (CoP_EA//d).astype(int)

cop_ea_up = CoP_EA//d + 1
cop_ea_up[cop_ea_up > cop_ea_mid.max()] = 0

cop_ea_down = CoP_EA//d - 1
cop_ea_down[cop_ea_down < 0] = cop_ea_mid.max()

cop_dm_mid = CoP_DM//d

cop_dm_up = CoP_DM//d + 1
cop_dm_up[cop_dm_up > cop_ea_mid.max()] = 0

cop_dm_down = CoP_DM//d - 1
cop_dm_down[cop_dm_down < 0] = cop_ea_mid.max()

del(CoP_DM,CoP_EA)




## filter dm particles by nearby neighbours for a given cell
#
#from itertools import compress
#
#
#i = 0  # choose a halo
## would alternatively choose a grid cell here, then loop over all halos within that cell
## only need the filtered, bound EA particles
#
## filter for nearby dm haloes
#nearby_dm = (((cop_ea_mid[i] == cop_dm_mid) | (cop_ea_mid[i] == cop_dm_up) | (cop_ea_mid[i] == cop_dm_down)).sum(axis=1) == 3))
#
## find their particles
#near_particles_DM = list(compress(particles_DM, nearby_dm)
#
#
## for a given EA halo ...
## loop over DM haloes
#for j in range(np.size(M_DM[local_dm])):
#
#    if(sum(np.in1d(particleIDs_DM[offset_DM[j] : offset_DM[j] + length_DM[j]],bound_particles_EA[i],assume_unique=True)) >= fracToFind * IDsToMatch):
#        print('matched: '+str(j))
#
#        reversed_halo_IDs = particleIDs_DM[offset_DM[j] : offset_DM[j]+length_DM[j]][0:IDsToMatch]
#
#        if sum(np.in1d(bound_particles_EA[i], reversed_halo_IDs, assume_unique=True)) >= fracToFind * IDsToMatch:
#            print('reversed match: '+str(j))
#
#        break
#





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


# 
# output = []
# output.append("# All halos \n")
# output.append("# box=%4.2fMpc z=%3.2f rho_crit=%5.3f epsilon=%4.3f IDsToMatch=%d\n"%(physicalBoxSize, redshift, rho_crit, softening, IDsToMatch))
# output.append("# Gr_EAGLE    Sub_EAGLE    Gr_DMONLY    Sub_DMONLY    frac_1    frac_2    M_EAGLE [Msun]    M_DMONLY [Msun]\n")


output = [] # np.zeros((len(M_EA),10))
# numb = np.size(M_EA)/jobs

matched_DM = []

# Loop over eagle halos
for n,i in enumerate(range(rank, np.size(M_EA), jobs)):
# for i in np.arange(len(M_EA)):

    print(np.round((float(i)/len(M_EA)),4) * 100,'% complete')
    sys.stdout.flush()

    # Consider only halos that are big enough
    #if M_EA[i] > massLimit and lengthType_EA[i,1] > IDsToMatch:
    print("Finding a match for halo (", Grp_EA[i], ",", Sub_EA[i], ") M=", M_EA[i])

    # Select 50 most bound particles of this halo
#    halo_IDs = particleIDs_EA[offset_EA[i] : offset_EA[i]+length_EA[i]]
#    mostBound_halo_IDs = halo_IDs[ halo_IDs % 2 == 0 ][0:IDsToMatch]

    # filter DM particles
    dm_list = np.where((M_EA[i] < mlc.mass_diff * M_DM) * (M_EA[i] > (1. / mlc.mass_diff) * M_DM ) * (((cop_ea_mid[i] == cop_dm_mid) | (cop_ea_mid[i] == cop_dm_up) | (cop_ea_mid[i] == cop_dm_down)).sum(axis=1) == 3))[0]

    dm_list = np.delete(dm_list,matched_DM)

    if len(dm_list) == 0: next

    for j in dm_list:
        # Select particles in this halo
        #thisHalo_IDs = particleIDs_DM[offset_DM[j] : offset_DM[j] + length_DM[j]]

        # Check whether the IDs from i are in j
        mask = np.in1d(particles_DM[j], bound_particles_EA[i], assume_unique=True)
        count = sum(mask)

        # Have we found enough particles ?
        if count >= mlc.fracToFind * mlc.IDsToMatch:
            print("Matched halo (", Grp_EA[i], ",", Sub_EA[i], ") to halo (", Grp_DM[j], ",", Sub_DM[j], ") M=", M_DM[j])#, "CoP", CoP_DM[j,:]

            match_fraction_EA = (float)(count) / (mlc.IDsToMatch)

            # print "Testing reversed match"
            #! does reversed match need to be done on *most bound* particles? ?????????
            #reversed_halo_IDs = particleIDs_DM[offset_DM[j] : offset_DM[j]+length_DM[j]][0:IDsToMatch]

            # Check whether the IDs from i are in j
            reversed_mask = np.in1d(particles_EA[i], particles_DM[j][0:mlc.IDsToMatch], assume_unique=True)
            reversed_count = sum(reversed_mask)


            if reversed_count >= mlc.fracToFind * mlc.IDsToMatch:

                match_fraction_DM = (float)(reversed_count) / (mlc.IDsToMatch)

                print("Match confirmed. Fractions:", match_fraction_EA, match_fraction_DM)

                # Print pair to file
                # output.append("%d %d %d %d %f %f %e %e %i %i\n"%( Grp_EA[i], Sub_EA[i], 
                #        Grp_DM[j], Sub_DM[j], match_fraction_EA, match_fraction_DM, 
                #        M_EA[i], M_DM[j], int(i), int(j)))

                # output[i,:] = np.array([Grp_EA[i], Sub_EA[i], Grp_DM[j], Sub_DM[j], 
                #                         match_fraction_EA, match_fraction_DM, 
                #                         M_EA[i], M_DM[j], int(i), int(j)])
                _out = {
                        'Grp_EA': Grp_EA[i], 
                        'Sub_EA': Sub_EA[i], 
                        'Grp_DM': Grp_DM[j], 
                        'Sub_DM': Sub_DM[j],
                        'match_fraction_EA': match_fraction_EA, 
                        'match_fraction_DM': match_fraction_DM, 
                        'M_EA': M_EA[i], 
                        'M_DM': M_DM[j], 
                        'i': int(i), 
                        'j': int(j)
                }
                output.append(_out)

                matched_DM.append(j)


            else:
                print("Match not confirmed.")
            break


import pandas as pd 
_df = pd.DataFrame(output)
_df.to_csv(output_folder+"matchedHalosSub_%s_z%dp%d_%d.dat"%(mlc.sim_name, z_int, z_dec, rank))


# file = open(, 'w')
# file.write(output)
# file.close()
