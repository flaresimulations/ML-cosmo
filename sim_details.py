import configparser
import numpy as np

class mlcosmo:

    def __init__(self, ini='config/config_L0012N0376.ini'):
        """
        ini: config file string
        """
        
        config = configparser.ConfigParser()
        config.read(ini)

        ## Sim details
        self.sim_name = config['Simulation details']['sim_name']
        self.sim_directory = config['Simulation details']['sim_directory']
        self.sim_hydro = self.sim_directory + config['Simulation details']['sim_hydro']
        self.sim_dmo = self.sim_directory + config['Simulation details']['sim_dmo']
        self.tag = config['Simulation details']['tag']

        ## match parameters
        self.fracToFind = float(config['Match parameters']['fracToFind'])
        self.IDsToMatch = int(config['Match parameters']['IDsToMatch'])
        self.massLimit = float(config['Match parameters']['massLimit'])
        self.mass_diff = float(config['Match parameters']['mass_diff'])       # mass ratio
        self.max_distance = float(config['Match parameters']['max_distance']) # kpc

        ## constants
        self.G = float(config['Constants']['G']) # kpc (1e10 Mo)^-1 km^2 s^-2
        self.unitMass = float(config['Constants']['unitMass'])
        self.unitLength = float(config['Constants']['unitLength'])
        self.Mpc = float(config['Constants']['Mpc'])


