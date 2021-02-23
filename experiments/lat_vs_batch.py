import pdb
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import sys

gpus = ['t4', 'p100', 'v100'] 
GPUs = ['T4', 'P100', 'V100']
#idle_pwr = [27, 60, 15]

fig, axs = plt.subplots(1, 2, figsize=(8,3.5), gridspec_kw={'hspace': 0, 'wspace': 0.3, 'top': 0.9, 'left':0.08, 'right':0.99, 'bottom':0.08})
width = 0.4

batches = list(range(1, 11))
# first plot throughput across different gpus
for i, gpu in enumerate(gpus):
    tail_list = []
    for batch_size in batches:
        path = f'logs/time_records/{gpu}_{batch_size}.json'
        with open(path) as f:
            lats = json.load(f)
        lats = lats[5:]
        tail_list.append(np.percentile(lats,95))
    axs[0].plot(batches, tail_list, label=gpu)

axs[0].set_xticks(batches)
axs[0].legend()
axs[0].set_title('tail latencies vs batch', fontsize=14)
axs[0].set_ylabel('tail latency\n(ms)', fontsize=13)
axs[0].set_xlabel('batch size', fontsize=13)

#axs[1].bar(x, qps_list, width=width) #TODO
#axs[1].set_xticks(x)
#axs[1].set_xticklabels(GPUs)
#axs[1].set_title('inference throughput', fontsize=14)
#axs[1].set_ylabel('throughput\n(query-per-second)', fontsize=13)
#
column = ' power.draw [W]'
for i, gpu in enumerate(gpus):
    energy_list = []
    path = f'logs/{gpu}.csv'
    df = pandas.read_csv(path)
    pwr = np.mean(df[column]) #- idle_pwr[i] #watt

    for batch_size in batches:
    time = lat_list[i]/1000 #second
    energy_list.append(pwr*time)
#
#axs[2].bar(x, energy_list, width)
#axs[2].set_xticks(x)
#axs[2].set_xticklabels(GPUs)
#axs[2].set_title('inference energy', fontsize=14)
#axs[2].set_ylabel('Energy (Joule)\nper query', fontsize=13)
#
#for ax in axs:
#    ax.grid(which='major', axis='y', ls='dotted')

plt.savefig(f'plots/lat_vs_batch.png')

