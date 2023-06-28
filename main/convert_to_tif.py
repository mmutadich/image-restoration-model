''''''
import numpy as np
import javabridge
import bioformats
import matplotlib
import matplotlib.pyplot as plt
from tifffile import imsave
import pickle

javabridge.start_vm(class_path=bioformats.JARS)

image_Data = bioformats.load_image('pGRS88_sample_3_multipoint.nd2',series= 0)
for i in range(0, 17):
    with open('planes/file_'+str(i+1)+'.tif', 'wb') as file:
        pickle.dump(image_Data[:,:,i], file)

#plt.imshow(image_1, interpolation='nearest')
#plt.show()

#print(type(image_0))
#imsave('image_0.tif',image_0)

javabridge.kill_vm()