import sys
sys.path.append('..') # Add src to path
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
from matplotlib.animation import FuncAnimation
from IPython.display import Image

#import tensorflow as tf
#
#from src.display import get_cmap
#from src.utils import make_log_dir
#
## comment these out if you don't have cartopy
#import cartopy.feature as cfeature
#from src.display.cartopy import make_ccrs,make_animation
#
#from make_dataset import NowcastGenerator,get_nowcast_train_generator,get_nowcast_test_generator
#
#from unet_benchmark import create_model
#from unet_benchmark import nowcast_mae, nowcast_mse


data_path="../experiments"
# Target locations of sample training & testing data
DEST_TRAIN_FILE= os.path.join(data_path,'data/processed/nowcast_training_000.h5')
DEST_TRAIN_META=os.path.join(data_path, 'data/processed/nowcast_training_000_META.csv')
DEST_TEST_FILE= os.path.join(data_path, 'data/processed/nowcast_testing_000.h5')
DEST_TEST_META= os.path.join(data_path, 'data/processed/nowcast_testing_000_META.csv')

# THIS DOWNLOADS APPROXIMATELY 40 GB DATASETS (AFTER DECOMPRESSION)
import boto3
from botocore.handlers import disable_signing
import tarfile
resource = boto3.resource('s3')
resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket=resource.Bucket('sevir')

print('Dowloading sample testing data')
if not os.path.exists(DEST_TEST_FILE):
    bucket.download_file('data/processed/nowcast_testing_000.h5.tar.gz',DEST_TEST_FILE+'.tar.gz')
    bucket.download_file('data/processed/nowcast_testing_000_META.csv',DEST_TEST_META)
    with tarfile.open(DEST_TEST_FILE+'.tar.gz') as tfile:
        tfile.extract('data/processed/nowcast_testing_000.h5','../experiments')
else:
    print('Test file %s already exists' % DEST_TEST_FILE)

#print('Dowloading sample training data')
#if not os.path.exists(DEST_TRAIN_FILE):
#    bucket.download_file('data/processed/nowcast_training_000.h5.tar.gz',DEST_TRAIN_FILE+'.tar.gz')
#    bucket.download_file('data/processed/nowcast_training_000_META.csv',DEST_TRAIN_META)
#    with tarfile.open(DEST_TRAIN_FILE+'.tar.gz') as tfile:
#        tfile.extract('data/processed/nowcast_training_000.h5','/scratch/li.baol/SEVIR')
#else:
#    print('Train file %s already exists' % DEST_TRAIN_FILE)
    
    
