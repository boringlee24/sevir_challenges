import pdb
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas

gpus = ['k40m', 'k80', 'p100', 'v100']
GPUs = ['K40m', 'K80', 'P100', 'V100']

fig, axs = plt.subplots(1, 2, figsize=(8,3.5), gridspec_kw={'hspace': 0, 'wspace': 0.3, 'top': 0.9, 'left':0.08, 'right':0.99, 'bottom':0.08})
x = np.arange(len(gpus))
width = 0.4

qps_list = []
lat_list = []
tail_list = []
# first plot throughput across different gpus
for i, gpu in enumerate(gpus):
    path = f'logs/time_records/{gpu}.json'
    with open(path) as f:
        lats = json.load(f)
    lat_mean = np.mean(lats)
    lat_list.append(lat_mean)
    tail_list.append(np.percentile(lats,90))
    qps = 1000 / lat_mean
    qps_list.append(qps)

axs[0].bar(x, tail_list, width=width) #TODO
axs[0].set_xticks(x)
axs[0].set_xticklabels(GPUs)
axs[0].set_title('inference throughput', fontsize=14)
axs[0].set_ylabel('throughput\n(query-per-second)', fontsize=13)

energy_list = []
column = ' power.draw [W]'
for i, gpu in enumerate(gpus):
    path = f'logs/{gpu}.csv'
    df = pandas.read_csv(path)
    pwr = np.mean(df[column]) #watt
    time = lat_list[i]/1000 #second
    energy_list.append(pwr*time)

axs[1].bar(x, energy_list, width)
axs[1].set_xticks(x)
axs[1].set_xticklabels(GPUs)
axs[1].set_title('inference energy', fontsize=14)
axs[1].set_ylabel('Energy (Joule)\nper query', fontsize=13)

for ax in axs:
    ax.grid(which='major', axis='y', ls='dotted')

plt.savefig('plots/fig1.png')

