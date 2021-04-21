import numpy as np

import matplotlib.pyplot as plt

from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import wp# , tpcf

import eagle_IO.eagle_IO as E

from sim_details import mlcosmo


mlc = mlcosmo(ini='config/config_cosma_L0100N1504.ini')
scale_factor = 0.908563; h = 0.6777
nthr = 4
coods = E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag,
                     "Subhalo/CentreOfPotential",
                     numThreads=nthr, noH=False, physicalUnits=False)

mstar = np.log10(E.read_array("SUBFIND", mlc.sim_hydro, mlc.tag,
                     "Subhalo/ApertureMeasurements/Mass/030kpc",
                     numThreads=nthr, noH=True)[:,4] * mlc.unitMass * h**2)

## obs data
fnames = [#'obs_data/farrow15-8.5-mass-9.5-2.00E-02-z-0.14-wprp.dat',
          'obs_data/farrow15-9.5-mass-10.0-2.00E-02-z-0.14-wprp.dat',  
          'obs_data/farrow15-10.0-mass-10.5-2.00E-02-z-0.14-wprp.dat',
          'obs_data/farrow15-10.5-mass-11.0-2.00E-02-z-0.14-wprp.dat',
          'obs_data/farrow15-11.0-mass-11.5-0.24-z-0.35-wprp.dat']

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(15,5))

for ax,lim,fname in zip([ax1,ax2,ax3,ax4], [9.5, 10, 10.5, 11], fnames):
    
    pi_max = 20.; Lbox = 100 * h #* scale_factor
    rp_binlims = np.logspace(-1.5,1.1,14)
    rp_bins = np.logspace(-1.4,1.0,13)

    mask = (mstar > lim) & (mstar < lim+0.5)   
    print("N_gals:", np.sum(mask))
    all_positions = return_xyz_formatted_array(coods[mask,0], coods[mask,1], coods[mask,2])
    wp_all = wp(all_positions, rp_binlims, pi_max, period=Lbox, num_threads='max')

    wp_tiles = np.zeros((8,len(rp_bins)))
    ## calculate jack knife errors
    for i,(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi) in \
            enumerate(zip([0, 0, Lbox/2, Lbox/2, 0, 0, Lbox/2, Lbox/2],
                [Lbox/2, Lbox/2, Lbox, Lbox, Lbox/2, Lbox/2, Lbox, Lbox],
                [0, Lbox/2, Lbox/2, 0, 0, Lbox/2, Lbox/2, 0],
                [Lbox/2, Lbox, Lbox, Lbox/2, Lbox/2, Lbox, Lbox, Lbox/2],
                [0, 0, 0, 0, Lbox/2, Lbox/2, Lbox/2, Lbox/2],
                [Lbox/2, Lbox/2, Lbox/2, Lbox/2, Lbox, Lbox, Lbox, Lbox])):
        
        mask = (mstar > lim) & (mstar < lim+0.5)
        mask = mask & np.invert((coods[:,0] > x_lo) & (coods[:,0] < x_hi) &\
                                (coods[:,1] > y_lo) & (coods[:,1] < y_hi) &\
                                (coods[:,2] > z_lo) & (coods[:,2] < z_hi))

        all_positions = return_xyz_formatted_array(coods[mask,0], coods[mask,1], coods[mask,2])

        wp_tiles[i] = wp(all_positions, rp_binlims, pi_max, period=Lbox, num_threads='max')


    sigma = np.sqrt(np.sum((wp_all - wp_tiles)**2, axis=0) \
            * (len(wp_tiles) - 1)/len(wp_tiles)) / rp_bins

    _y = wp_all / rp_bins
    err = ([np.log10(_y) - np.log10(_y - sigma), np.log10(_y + sigma) - np.log10(_y)]) 

    ax.errorbar(np.log10(rp_bins), np.log10(wp_all / rp_bins), 
                yerr=err, label='Ref-100')

    _dat = np.loadtxt(fname)
    err = [np.log10(_dat[:,1]) - np.log10(_dat[:,1] - _dat[:,2]), 
            np.log10(_dat[:,1] + _dat[:,2]) - np.log10(_dat[:,1])]

    ax.errorbar(np.log10(_dat[:,0]), np.log10(_dat[:,1]/_dat[:,0]), yerr=err, label='GAMA')
   
    ax.text(0.1, 0.1, '$%.1f < \mathrm{log_{10}}(M_{\star} / M_{\odot} h^{-2}) < %.1f$'%(lim,lim+0.5), 
            transform=ax.transAxes)
    ax.set_xlim(-1.7,1.8); ax.set_ylim(-3,5)

ax1.legend()
plt.show()
