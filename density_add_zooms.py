import pandas as pd
import numpy as np

from sim_details import mlcosmo

for i in np.arange(30):
    mlc = mlcosmo(ini='config/config_CE-%i.ini'%i)
    data = pd.read_csv(f'output/{mlc.sim_name}_{mlc.tag}_match.csv')

    data['Density_R1'] = np.loadtxt(f'output/density_{mlc.sim_name}_{mlc.tag}_R1.txt')
    data['Density_R2'] = np.loadtxt(f'output/density_{mlc.sim_name}_{mlc.tag}_R2.txt')
    data['Density_R4'] = np.loadtxt(f'output/density_{mlc.sim_name}_{mlc.tag}_R4.txt')
    data['Density_R8'] = np.loadtxt(f'output/density_{mlc.sim_name}_{mlc.tag}_R8.txt')
    # data['Density_R16'] = np.loadtxt(f'output/density_{mlc.sim_name}_{mlc.tag}_R16.txt')

    data.to_csv(f'output/{mlc.sim_name}_{mlc.tag}_match.csv')

