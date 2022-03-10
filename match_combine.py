import sys
import os
import glob
import pandas as pd

# from sim_details import mlcosmo
# _config = str(sys.argv[1])
# mlc = mlcosmo(ini=_config)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config", help="config file", type=str)
parser.add_argument("--region", help="if a flares zoom region, provide the region number",
                    type=int, default=None)
parser.add_argument("--tag", help="snapshot tag string", type=str, default=None)
args = parser.parse_args()

from sim_details import mlcosmo
mlc = mlcosmo(ini=args.config, region=args.region, tag=args.tag)


print("==========\nSim: %s\nTag: %s\n===========\n"%(mlc.sim_name,mlc.tag))

output = 'output/'

_file = "%s/matchedHalosSub_%s_%s.*.dat"%(output,mlc.sim_name, mlc.tag)
print(_file)

files = sorted(glob.glob(_file))
match = [None] * len(files)
for i,f in enumerate(files): match[i] = pd.read_csv(f)
match = pd.concat(match)
match.reset_index(inplace=True)

match.to_csv('%s/matchedHalosSub_%s_%s.dat'%(output,mlc.sim_name,mlc.tag))

for i,f in enumerate(files): os.remove(f)

