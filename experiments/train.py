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

data_path="/scratch/li.baol/SEVIR"
# Target locations of sample training & testing data
DEST_TRAIN_FILE= os.path.join(data_path,'data/processed/nowcast_training_000.h5')
DEST_TRAIN_META=os.path.join(data_path, 'data/processed/nowcast_training_000_META.csv')
DEST_TEST_FILE= os.path.join(data_path, 'data/processed/nowcast_testing_000.h5')
DEST_TEST_META= os.path.join(data_path, 'data/processed/nowcast_testing_000_META.csv')

# Control how many samples are read.   Set to -1 to read all 5000 samples.
N_TRAIN=-1
TRAIN_VAL_FRAC=0.8
#set_trace()
N_TEST=-1

# Loading data takes a few minutes
with h5py.File(DEST_TRAIN_FILE,'r') as hf:
    Nr = N_TRAIN if N_TRAIN>=0 else hf['IN_vil'].shape[0]
    X_train = hf['IN_vil'][:Nr]
    Y_train = hf['OUT_vil'][:Nr]
    training_meta = pd.read_csv(DEST_TRAIN_META).iloc[:Nr]
    X_train,X_val=np.split(X_train,[int(TRAIN_VAL_FRAC*Nr)])
    Y_train,Y_val=np.split(Y_train,[int(TRAIN_VAL_FRAC*Nr)])
    training_meta,val_meta=np.split(training_meta,[int(TRAIN_VAL_FRAC*Nr)])
#set_trace()       
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

#unet.summary()
#pdb.set_trace()
exprmt_dir='logs'
make_log_dir = 'plots'
Path(exprmt_dir).mkdir(parents=True,exist_ok=True)
Path(make_log_dir).mkdir(parents=True,exist_ok=True)

opt=params['opt'](learning_rate=params['learning_rate'])
unet.compile(optimizer=opt, loss=params['loss'],loss_weights=[params['loss_weights']])

# Training 10 epochs takes around 10-20 minutes on GPU
num_epochs=params['num_epochs']
batch_size=params['batch_size']

callbacks=[
    tf.keras.callbacks.ModelCheckpoint(exprmt_dir+'/unet.hdf5', 
                    monitor='val_loss',save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=exprmt_dir+'/tboardlogs')
]

history = unet.fit(x=X_train, y=Y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                   callbacks=callbacks,
                  validation_data=(X_val, Y_val))

plt.plot(history.history['loss'],label='Train loss')
plt.plot(history.history['val_loss'],label='Val loss')
plt.legend()
plt.savefig('plots/train.png')
