from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np

import sys
import os
import glob
from tifffile import imread
from csbdeep.utils import axes_dict, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.data import RawData, create_patches
from csbdeep.data import no_background_patches, norm_percentiles, sample_percentiles
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

dataFiles = [f for f in os.listdir() if f.startswith('data')]
print(dataFiles)
for path in dataFiles:
    patches_path =  os.path.join(path, 'my_training_data.npz')
    raw_data_path = os.path.join(path, 'train')
    model_name = 'model_'+str(path)
    print(patches_path)
    print(raw_data_path)
    print(model_name)

    #patches_path = 'data8/my_training_data.npz'
    #raw_data_path = 'data8/train'
    #model_name = 'model8'

    '''GENERATING DATA
    The data must be saved as pairs of images with the same name in two different
    folders (GT and low)

    TO DO: change the hard coded files to be an input in batch file
    '''
    raw_data = RawData.from_folder (
        basepath    = raw_data_path,
        source_dirs = ['low'],
        target_dir  = 'GT',
        axes        = 'YX',
    )

    X, Y, XY_axes = create_patches (
        raw_data            = raw_data,
        patch_size          = (128,128),
        patch_filter        = no_background_patches(0),
        n_patches_per_image = 2,
        save_file           = patches_path,
    )

    assert X.shape == Y.shape
    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)

    #TRAINING
    (X,Y), (X_val,Y_val), axes = load_training_data(patches_path, validation_split=0.05, verbose=True)
    #change validation_split based on number of images

    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

    #Initialise model
    config = Config(axes, n_channel_in, n_channel_out, unet_kern_size=3, train_batch_size=8, train_steps_per_epoch=400) #increase to 400 once stopped messing around
    print(config)
    vars(config)

    model = CARE(config, model_name, basedir='models') #make bash input

    model.keras_model.summary()

    history = model.train(X,Y, validation_data=(X_val,Y_val))

    print(sorted(list(history.history.keys())))
