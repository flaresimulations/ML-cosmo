import numpy as np

baldry_12 = {}

baldry_12['logM'] = np.array([6.25,6.75,7.10,7.30,7.50,7.70,7.90,8.10,8.30,
          8.50, 8.70, 8.90, 9.10, 9.30, 9.50, 9.70, 9.90, 10.10, 
          10.30, 10.50, 10.70, 10.90, 11.10, 11.30, 11.50, 11.70, 11.90])

baldry_12['bin_width'] = np.array([0.50,0.50,0.20,0.20,0.20,0.20,0.20,0.20,
          0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.20,
          0.20 ,0.20 ,0.20 ,0.20 ,0.20 ,0.20 ,0.20])

baldry_12['phi'] = np.array([31.1,18.1,17.9,43.1,31.6,34.8,27.3,28.3,23.5,
          19.2,18.0,14.3,10.2,9.59,7.42,6.21,5.71,5.51,5.48,5.12,3.55,2.41, 
          1.27,0.338,0.042,0.021,0.042]) * 1e-3

baldry_12['err'] = np.array([21.6,6.6,5.7,8.7,9.0,8.4,4.2,2.8,3.0,1.2,2.6, 
          1.7, 0.6  ,0.55 ,0.41 ,0.37 ,0.35 ,0.34 ,0.34 ,0.33 ,0.27 ,0.23, 
          0.16, 0.085 ,0.030 ,0.021 ,0.030]) * 1e-3

baldry_12['N'] = np.array([9,19,18,46,51,88,140,243,282,399,494,505,449,423,
              340,290,268,260,259,242,168,114,60,16,2,1,2])

