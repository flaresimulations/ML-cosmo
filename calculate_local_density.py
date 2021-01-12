import numpy as np
import math
import sys

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo
_config = str(sys.argv[1])
mlc = mlcosmo(ini=_config)

nthr=16
zoom=False

radii = [1,2,4,8,16]
volumes = [(4./3) * np.pi * r**3 for r in radii]

_idx = int(sys.argv[2])
R = radii[_idx]
V = volumes[_idx]

dR = 0.
Rmax = np.max(radii) + dR

output='output/'
_idx = np.loadtxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt', dtype=int)
idx_DM = _idx[:,1]


if zoom:
    print("do zoom stuff")
else:
    particle_pos = E.read_array("SNAPSHOT", mlc.sim_dmo, mlc.tag, 
                                "PartType1/Coordinates", numThreads=nthr, noH=True)

    dm_pmass = E.read_header("SNAPSHOT", mlc.sim_dmo, mlc.tag, "MassTable")[1]

    CoP = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, 
                 "Subhalo/CentreOfPotential", numThreads=nthr, noH=True)[idx_DM] 

    _fact = 1e-2
    mask = np.random.rand(len(particle_pos)) < _fact
    _tree = cKDTree(particle_pos[mask], boxsize=100) 

    # density_output = {r: np.zeros(len(CoP)) for r in radii}
    _out = np.zeros(len(CoP))
    step=1000
    
    # for R,V in zip(radii,volumes):
    print("Radius: %02d Mpc"%R)
    for i in range(0, len(CoP), step):
        cop_tree = cKDTree(CoP[i:i+step], boxsize=100)
        N_part = cop_tree.query_ball_tree(_tree,r=R)
        _out[i:i+step] = [len(_t) for _t in N_part]
        
    _out *= dm_pmass * (1./_fact) * (1./V)

np.savetxt(output + 'density_' + mlc.sim_name + '_' + mlc.tag + '_R' + str(R) + ".txt",_out)

# eagle = pd.read_csv((output + mlc.sim_name + '_' + mlc.tag + "_match.csv"))
# 
# for R in radii:
#     eagle['density_%02d_Mpc'] = density_output[R]
# 
# eagle.to_csv(output + mlc.sim_name + '_' + mlc.tag + "_match.csv")

