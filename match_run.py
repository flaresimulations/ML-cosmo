# This code was adapted from the python examples on the Eagle wiki given here:
# http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=eagle:documentation:reading_python&s[]=bound

import pickle

import pandas as pd 
import numpy as np
import math
import sys

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo
mlc = mlcosmo(ini='config/config_CE-0.ini')

output_folder = 'output/'

## serial arguments
# rank = 0
# jobs = 1
rank = int(sys.argv[1])  # rank of process
jobs = int(sys.argv[2])  # total number of processes

M_EA,M_DM,CoP_EA,CoP_DM,particles_DM,bound_particles_EA,\
   particles_EA,particles_DM,Grp_EA,Sub_EA,Grp_DM,Sub_DM = \
   pickle.load(open(output_folder + mlc.sim_name + "_match_data.p",'rb'))


# boxSize = E.read_header("PARTDATA", mlc.sim_hydro, mlc.tag, "BoxSize")
# hubbleParam = E.read_header("PARTDATA", mlc.sim_hydro, mlc.tag, "HubbleParam")
# physicalBoxSize = boxSize / hubbleParam
# Create rough grid of positions to reduce number of distance calculations
# rat = int(physicalBoxSize * mlc.unitLength / mlc.max_distance)
# d = physicalBoxSize * mlc.unitLength/rat # ~> 8000
d = 1000

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


output = [] 
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



_df = pd.DataFrame(output)
_df.to_csv(output_folder+"matchedHalosSub_%s_%s.%03d.dat"%(mlc.sim_name, mlc.tag, rank))


# file = open(, 'w')
# file.write(output)
# file.close()
