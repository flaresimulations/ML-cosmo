import numpy as np
import math
import sys

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo
_config = str(sys.argv[1])
mlc = mlcosmo(ini=_config)


zoom=bool(sys.argv[2])
boxsize=mlc.boxsize # float(sys.argv[2])
nthr=8

radii = [1,2,4,8]#,16][1,2
volumes = [(4./3) * np.pi * r**3 for r in radii]

# _idx = int(sys.argv[2])
# R = radii[_idx]
# V = volumes[_idx]

output='output/'
indexes = np.loadtxt(output + mlc.sim_name + '_' + mlc.tag + '_indexes.txt', dtype=int)
idx_EA = indexes[:,0]
idx_DM = indexes[:,1]
    
## ---- Run density calculation on high-res particles

particle_pos = E.read_array("SNAPSHOT", mlc.sim_dmo, mlc.tag, 
                            "PartType1/Coordinates", numThreads=nthr, noH=True)

dm_pmass = E.read_header("SNAPSHOT", mlc.sim_dmo, mlc.tag, "MassTable")[1]

CoP = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, 
             "Subhalo/CentreOfPotential", numThreads=nthr, noH=True)[idx_DM] 

shm = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, 
                   "Subhalo/Mass", numThreads=nthr, noH=True)[idx_DM]  * mlc.unitMass

# mcrit200 = E.read_array("SUBFIND", mlc.sim_dmo, mlc.tag, 
#                         "FOF/Group_M_Crit200", numThreads=nthr)[idx_DM] * mlc.unitMass

# mstar = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag, 
#                  "Subhalo/Stars/Mass", numThreads=nthr)[idx_EA] * mlc.unitMass

match_indexes = np.where((shm > 1e10))[0]

_fact = 2e-3
mask = np.random.rand(len(particle_pos)) < _fact
_tree = cKDTree(particle_pos[mask], boxsize=boxsize) 

del(particle_pos)

for R,V in zip(radii,volumes):
    print("Radius: %02d Mpc"%R)
    
    _out = np.zeros(len(CoP))
    step=1000
    for i in range(0, len(CoP), step):
        print(i)
        idxs = match_indexes[i:i+step]
        cop_tree = cKDTree(CoP[idxs], boxsize=boxsize)
        N_part = cop_tree.query_ball_tree(_tree,r=R)
        _out[idxs] = [len(_t) for _t in N_part]
    
    
    _out *= dm_pmass * mlc.unitMass * (1./_fact)
    
    if zoom:
        #boxsize=3200
        # cop_tree = cKDTree(CoP[idxs], boxsize=boxsize)
        cop_tree = cKDTree(CoP, boxsize=boxsize)
    
        p2_pos = E.read_array("SNAPSHOT", mlc.sim_dmo, mlc.tag, 
                              "PartType2/Coordinates", numThreads=nthr, noH=True)  
        p3_pos = E.read_array("SNAPSHOT", mlc.sim_dmo, mlc.tag, 
                              "PartType3/Coordinates", numThreads=nthr, noH=True)  
        p2_mass = E.read_array("SNAPSHOT", mlc.sim_dmo, mlc.tag, 
                               "PartType2/Mass", numThreads=nthr, noH=True)  
        p3_mass = E.read_array("SNAPSHOT", mlc.sim_dmo, mlc.tag, 
                               "PartType3/Mass", numThreads=nthr, noH=True)  
        
        _tree2 = cKDTree(p2_pos, boxsize=boxsize) 
        del(p2_pos)
        N_part = cop_tree.query_ball_tree(_tree2,r=R)
        
        sum_p2_mass = np.zeros(len(match_indexes))
        for i,idxs in enumerate(N_part):
            if len(idxs) > 0:
                sum_p2_mass[i] =  np.sum(p2_mass[np.array(idxs)]) * 1e10
        
        _tree3 = cKDTree(p3_pos, boxsize=boxsize) 
        del(p3_pos)
        N_part = cop_tree.query_ball_tree(_tree3,r=R)
        
        sum_p3_mass = np.zeros(len(match_indexes))
        for i,idxs in enumerate(N_part):
            if len(idxs) > 0:
                sum_p3_mass[i] =  np.sum(p3_mass[np.array(idxs)]) * 1e10
    
        _out[match_indexes] += sum_p2_mass + sum_p3_mass
    
    
    _out *= (1./V)
    
    np.savetxt(output + 'density_' + mlc.sim_name + '_' + mlc.tag + '_R' + str(R) + ".txt",_out)
    
