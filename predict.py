import os
import sys
import numpy as np
from csbdeep.models import CARE
from tifffile import imread, imwrite

dataFiles = [f for f in os.listdir() if f.startswith('data')]
print(dataFiles)
for path in dataFiles:
    print(path)
    dir_low =  os.path.join(path, 'test/low')
    dir_result = os.path.join(path, 'test/predict')
    model_name = 'model_'+str(path)
    print(dir_low)
    print(dir_result)
    print(model_name)
    #dir_low = 'data10/test/low'
    #dir_result = 'data10/test/predict/'
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)

    low_SNR_files = [f for f in os.listdir(dir_low) if f.endswith('.tif')]
    if len(low_SNR_files) == 0:
        raise Exception('No .tif images found in ' + dir_low)

    model = CARE(config=None, name=model_name, basedir='models')

    axes = 'YX'
    for i in range (len(low_SNR_files)):
        image = imread(os.path.join(dir_low, low_SNR_files[i]))
        restored = model.predict(image, axes)
        print(type(restored))
        img = np.array(restored, 'uint8')
        #split tif around tif to avoid errors
        imwrite(dir_result+low_SNR_files[i]+'_predict.tif', img, photometric='minisblack')
