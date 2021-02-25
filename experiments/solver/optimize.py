from scipy.optimize import linprog
import json
import pdb
import numpy as np

with open('config.json') as f:
    inputs = json.load(f)

throughput = [-x for x in inputs['throughput_qps']]
target_lat = inputs['target_lat']
latency = [target_lat - x for x in inputs['latency_ms']]
energy = inputs['energy_Jpq']

############ optimization kernel ###############

obj = energy
lhs_ineq = [throughput, latency]
rhs_ineq = [-inputs['target_qps'], 0]
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, method='revised simplex')
optima = np.ceil(opt.x)

###############################################

hardware = inputs['hardware']
energy_opt = np.multiply(np.multiply(throughput,energy), optima)
energy_opt = -sum(energy_opt)
print(f'optimal energy: {energy_opt} (Joule) per second')
for i, item in enumerate(hardware):
    print(f'Use {int(optima[i])} {item}')


