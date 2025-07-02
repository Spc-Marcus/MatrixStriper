import os
os.environ['GRB_WLSACCESSID'] = 'af4b8280-70cd-47bc-aeef-69ecf14ecd10'
os.environ['GRB_WLSSECRET'] = '04da6102-8eb3-4e38-ba06-660ea8f87bf2'
os.environ['GRB_LICENSEID'] = '2669217'
import gurobipy as grb
print(grb.gurobi.version())
m = grb.Model()
print(m.getAttr('LicenseExpiration'))