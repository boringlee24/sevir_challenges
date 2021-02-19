import sys
sys.path.append('..') # Add src to path
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from src.display import get_cmap
from src.utils import make_log_dir
# comment these out if you don't have cartopy
import cartopy.feature as cfeature
from src.display.cartopy import make_ccrs,make_animation
from make_dataset import NowcastGenerator,get_nowcast_train_generator,get_nowcast_test_generator
from unet_benchmark import create_model
from unet_benchmark import nowcast_mae, nowcast_mse
import pdb
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import time
import sys

data_path="/scratch/li.baol/SEVIR"
# Target locations of sample training & testing data
DEST_TEST_FILE= os.path.join(data_path, 'data/processed/nowcast_testing_000.h5')
DEST_TEST_META= os.path.join(data_path, 'data/processed/nowcast_testing_000_META.csv')

# Control how many samples are read.   Set to -1 to read all 5000 samples.
N_TRAIN=-1
TRAIN_VAL_FRAC=0.8
#set_trace()
N_TEST=-1
num_iters = 10000
wait = True # set to False when measuring power, TODO

#for i in range(num_iters):
#    print(f'{i}/{num_iters} progress', end='\r', flush=True)
#    time.sleep(0.1)
#pdb.set_trace()

with h5py.File(DEST_TEST_FILE,'r') as hf:
    Nr = N_TEST if N_TEST>=0 else hf['IN_vil'].shape[0]
    X_test = hf['IN_vil'][:Nr]
    Y_test = hf['OUT_vil'][:Nr]
    testing_meta=pd.read_csv(DEST_TEST_META).iloc[:Nr]

# Add more as needed
params={
    'start_neurons'   :16,      # Controls size of hidden layers in CNN, higher = more complexity 
    'activation'      :'relu',  # Activation used throughout the U-Net,  see https://www.tensorflow.org/api_docs/python/tf/keras/activations
    'loss'            :'mae',   # Either 'mae' or 'mse', or others as https://www.tensorflow.org/api_docs/python/tf/keras/losses
    'loss_weights'    :0.021,    # Scale for loss.  Recommend squaring this if using MSE
    'opt'             :tf.keras.optimizers.Adam,  # optimizer, see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    'learning_rate'   :0.001,   # Learning rate for optimizer
    'num_epochs'      :10,       # Number of epochs to train for
    'batch_size'      :8        # Size of batches during training
}

unet = create_model(start_neurons=params['start_neurons'],activation=params['activation']) 
unet.load_weights('logs/unet.hdf5')

batch_size,batch_num=8,4
bs,be=batch_size*batch_num,batch_size*(batch_num+1)
x_test,y_test,meta = X_test[bs:be],Y_test[bs:be],testing_meta.iloc[bs:be]

# warm up
unet.predict(x_test)
time_record = []

for i in range(num_iters):
    indexs = np.random.choice(len(X_test), batch_size, replace=False)
    x_test = X_test[indexs]
    if len(x_test) != 8:
        pdb.set_trace()
    t_start = time.time()
    unet.predict(x_test)
    t_end = time.time()
    lat = round((t_end - t_start)*1000,3) 
    time_record.append(lat) # ms
    if wait:
        time.sleep(0.1)
        print(f'{i}/{num_iters} done, latency is {lat}ms', end='\r', flush=True)
if wait:
    gpu = sys.argv[1]
    with open(f'logs/time_records/{gpu}.json', 'w') as fp:
        json.dump(time_record, fp, indent=4)
