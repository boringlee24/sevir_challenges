import pdb
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import sys

batch_size = '10'
gpus = ['cpu', 'k80', 'm60', 't4d', 'p100', 'v100a'] # cpu node is c0191
#GPUs = ['CPU', 't4D', 'P100', 'V100A']
GPUs = gpus[:]
#idle_pwr = [27, 60, 15]

fig, axs = plt.subplots(1, 3, figsize=(12,3.5), gridspec_kw={'hspace': 0, 'wspace': 0.3, 'top': 0.9, 'left':0.08, 'right':0.99, 'bottom':0.08})
x = np.arange(len(gpus))
width = 0.4

qps_list = []
lat_list = []
tail_list = []
# first plot throughput across different gpus
for i, gpu in enumerate(gpus):
    if batch_size == '8':
        path = f'logs/time_records/{gpu}.json'
    else:
         path = f'logs/time_records/{gpu}_{batch_size}.json'
       
    with open(path) as f:
        lats = json.load(f)
    lats = lats[5:]
    lat_mean = np.mean(lats)
    lat_list.append(lat_mean)
    tail_list.append(np.percentile(lats,95))
    qps = 1000 / lat_mean
    qps_list.append(qps)

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
#        pdb.set_trace()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%s' % round(height,2),
                ha='center', va='bottom')


rect = axs[0].bar(x, lat_list, width=width) #TODO
axs[0].set_xticks(x)
axs[0].set_xticklabels(GPUs)
axs[0].set_title('inference mean latency', fontsize=14)
axs[0].set_ylabel('latency\n(ms)', fontsize=13)
autolabel(rect, axs[0])

rect = axs[1].bar(x, qps_list, width=width) #TODO
axs[1].set_xticks(x)
axs[1].set_xticklabels(GPUs)
axs[1].set_title('inference throughput', fontsize=14)
axs[1].set_ylabel('throughput\n(query-per-second)', fontsize=13)
autolabel(rect, axs[1])

energy_list = []
column = ' power.draw [W]'
for i, gpu in enumerate(gpus):
    if gpu == 'cpu':
        pwr = 48
    else:
        path = f'logs/{gpu}_{batch_size}.csv'
        df = pandas.read_csv(path)
#        pwr = np.mean(df[column]) #- idle_pwr[i] #watt
        pwr = np.percentile(df[column], 90) #TODO

    time = lat_list[i]/1000 #second
    energy_list.append(pwr*time)

rect = axs[2].bar(x, energy_list, width)
axs[2].set_xticks(x)
axs[2].set_xticklabels(GPUs)
axs[2].set_title('inference energy', fontsize=14)
axs[2].set_ylabel('Energy (Joule)\nper query', fontsize=13)
autolabel(rect, axs[2])

for ax in axs:
    ax.grid(which='major', axis='y', ls='dotted')


plt.savefig(f'plots/batch_{batch_size}.png')

