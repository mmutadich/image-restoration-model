import os
import bioformats
import javabridge
import numpy as np
from tifffile import imwrite

# pick channel to use for low SNR image
for whichChannel in range (10, 11):
    # make directory structure for training data
    dataDir = 'data'+str(whichChannel)+'/'
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
        os.makedirs(dataDir+'/train/')
        os.makedirs(dataDir+'/train/GT')
        os.makedirs(dataDir+'/train/Raw')
        os.makedirs(dataDir+'/test/')
        os.makedirs(dataDir+'/test/GT')
        os.makedirs(dataDir+'/test/Raw')

    # used to specify train-test split
    numImages = 126
    currImgNumber = 0
    nd2Path = 'nd2files'

    nd2Files = [f for f in os.listdir(nd2Path) if f.endswith('.nd2')]

    # start Java VM, adding bioformats.JARS to the class path
    javabridge.start_vm(class_path=bioformats.JARS)

    for img in nd2Files:
    # load image metadata
        imagePath = os.path.join(nd2Path, img)
        metadata = bioformats.get_omexml_metadata(path=imagePath)
        o = bioformats.OMEXML(metadata)
    # parse dimension information from metadata
        numPoints = o.get_image_count()
        for whichPoint in range(numPoints):
            # read the image
            # by default, reads all channels at specified z and t position
                with bioformats.ImageReader(path=imagePath) as reader:
                    img = reader.read(series=whichPoint, rescale=False, z=0, t=0)
                    reader.close()

                img = np.array(img, 'uint16')

            # implement train-test split
                if currImgNumber % 25 == 0:
                    outDir = dataDir+'test/'
                else:
                    outDir = dataDir + 'train/'

            # write ground truth and low SNR images
                imwrite(outDir+'GT/img'+str(currImgNumber)+'.tif',img[:, :, 1],photometric='minisblack')
                imwrite(outDir+'Raw/img'+str(currImgNumber)+'.tif',img[:, :, whichChannel],photometric='minisblack')

                currImgNumber += 1

# close the java VM
javabridge.kill_vm()
