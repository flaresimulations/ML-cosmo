# This code was adapted from the python examples on the Eagle wiki given here:
# http://eagle.strw.leidenuniv.nl/wiki/doku.php?id=eagle:documentation:reading_python&s[]=bound

import argparse
import math
import sys

import h5py
import pandas as pd 
import numpy as np

import eagle_IO.eagle_IO as E


parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file", type=str)
parser.add_argument("--region", help="if a flares zoom region, provide the region number",
                    type=int, default=None)
parser.add_argument("--tag", help="snapshot tag string", type=str, default=None)
parser.add_argument("--rank", help="processor rank", type=int, default=None)
parser.add_argument("--jobs", help="total number of jobs", type=int, default=None)
args = parser.parse_args()

from sim_details import mlcosmo
mlc = mlcosmo(ini=args.config, region=args.region, tag=args.tag)

output_folder = 'output/'

if args.rank is not None: rank = args.rank
else: rank = 0

if args.jobs is not None: jobs = args.jobs
else: jobs = 1


with h5py.File(output_folder + mlc.sim_name + '_' + mlc.tag + "_match_data.h5", 'r') as hf:
    M_EA = hf['M_EA'][:]
    M_DM = hf['M_DM'][:]
    CoP_EA = hf['CoP_EA'][:]
    CoP_DM = hf['CoP_DM'][:]
    bound_particles_EA = hf['bound_particles_EA'][:]
    Grp_EA = hf['Grp_EA'][:]
    Grp_DM = hf['Grp_DM'][:]
    Sub_EA = hf['Sub_EA'][:]
    Sub_DM = hf['Sub_DM'][:]
    particles_EA = hf['particles_EA'][:]
    particles_DM = hf['particles_DM'][:]


# M_EA,M_DM,CoP_EA,CoP_DM,bound_particles_EA,\
#    particles_EA,particles_DM,Grp_EA,Sub_EA,Grp_DM,Sub_DM = \
#    pickle.load(open(output_folder + mlc.sim_name + '_' + mlc.tag + "_match_data.p",'rb'))



cop_ea_mid = (CoP_EA//mlc.max_distance).astype(int)

cop_ea_up = CoP_EA//mlc.max_distance + 1
cop_ea_up[cop_ea_up > cop_ea_mid.max()] = 0

cop_ea_down = CoP_EA//mlc.max_distance - 1
cop_ea_down[cop_ea_down < 0] = cop_ea_mid.max()

cop_dm_mid = CoP_DM//mlc.max_distance

cop_dm_up = CoP_DM//mlc.max_distance + 1
cop_dm_up[cop_dm_up > cop_ea_mid.max()] = 0

cop_dm_down = CoP_DM//mlc.max_distance - 1
cop_dm_down[cop_dm_down < 0] = cop_ea_mid.max()

del(CoP_DM,CoP_EA)


output = [] 
matched_DM = []

for n,i in enumerate(range(rank, np.size(M_EA), jobs)):
    #if i%100 == 0: 
    print(np.round((float(i)/len(M_EA)),4) * 100,'% complete')
    sys.stdout.flush()

    _match_flag = False
    print("Finding a match for halo (", Grp_EA[i], ",", Sub_EA[i], 
          ") log10(M)=%.3f"%np.log10(M_EA[i]))

    # filter DM particles
    dm_list = np.where((M_EA[i] < mlc.mass_diff * M_DM) &\
                       (M_EA[i] > (1. / mlc.mass_diff) * M_DM ) &\
              (( (cop_ea_mid[i] == cop_dm_mid) |\
                 (cop_ea_mid[i] == cop_dm_up) |\
                 (cop_ea_mid[i] == cop_dm_down) ).sum(axis=1) == 3))[0]

    not_matched = np.where(~np.in1d(dm_list, matched_DM))[0]
    if len(not_matched) == 0:
        continue
    else:
        dm_list = np.array(dm_list)[not_matched]

    # dm_list = np.delete(dm_list,matched_DM)
    # if len(dm_list) == 0:
    #     print("No match found.")

    for j in dm_list:
        # Check whether the IDs from i are in j
        mask = np.in1d(particles_DM[j], bound_particles_EA[i], assume_unique=True)
        count = np.sum(mask)

        # Have we found enough particles ?
        if count >= mlc.fracToFind * mlc.IDsToMatch:
            print("Matched halo (", Grp_EA[i], ",", Sub_EA[i], ") to halo (", Grp_DM[j], ",", Sub_DM[j], ") M=", M_DM[j])#, "CoP", CoP_DM[j,:]

            match_fraction_EA = (float)(count) / (mlc.IDsToMatch)

            # Check whether the IDs from i are in j
            reversed_mask = np.in1d(particles_EA[i], particles_DM[j][0:mlc.IDsToMatch], 
                                    assume_unique=True)
            reversed_count = np.sum(reversed_mask)

            if reversed_count >= mlc.fracToFind * mlc.IDsToMatch:
                match_fraction_DM = float(reversed_count) / (mlc.IDsToMatch)
                print("Match confirmed. Fractions:", match_fraction_EA, match_fraction_DM)

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
                _match_flag = True
                break # continue

            else:
                print("Match not confirmed. Reversed count:", reversed_count)
            break
        else:
            continue
    
    if not _match_flag: print("No match found")


_df = pd.DataFrame(output)
_df.to_csv(output_folder+"matchedHalosSub_%s_%s.%03d.dat"%(mlc.sim_name, mlc.tag, rank))

