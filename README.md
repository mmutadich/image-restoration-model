## Denoising Three-Dimensional Bacterial Biofilm Images Using Machine Learning  
# Abstract
**Mia Mutadich**  
*Mentors: Dianne K. Newman, Georgia R. Squyres*<br>
*Summer 2023*

Over the last two decades, the development of high-resolution, three-dimensional imaging has been instrumental in unveiling the structural and dynamical complexities of biofilms, enabling researchers to better understand their development and functions. Biofilms are structured communities of cells that given their relevance to human health, industry, and the environment, are the subject of intense research across multiple disciplines. However, direct observation of individual cell behavior within biofilms has been limited by the low signal-to-noise (SNR) of fluorescence microscopy images. We are applying deep neural networks for image restoration to three-dimensional images of biofilms to reduce noise without blurring the underlying signal – a limitation of previous techniques that we aim to overcome. To develop a neural network for this task, we started with a simpler 2D data set with a known ground truth. Because SNR in 3D biofilms varies with depth, we represented this by acquiring 2D images at various SNR levels. We trained an existing neural network architecture on these images and wrote an algorithm to compare its performance on the full range of representative SNR values. Finally, after developing a method to simulate noise in three-dimensional biofilm images, we aim to be able to retrain our optimized neural network on our semi-synthetic dataset so that it can denoise three-dimensional images. 

# Method 
The first aim of my project is to create and train a neural network to denoise a sample two-dimensional data set; this trained neural network will later be retrained on a three-dimensional dataset to be able to denoise three-dimensional images. My mentor Dr. Georgia Squyres provided me with a data set that she collected of two-dimensional images of bacteria but at different exposure times to simulate the different signal intensities that images at different depths of the biofilm would have when imaged – this data has been used to train my neural network. The way that we quantitively characterize the level of noise in an image is by calculating the ratio of the number of signal pixels, where the signal is the bacteria in the image, and the number of noise pixels, where the noise is the background - this is called the signal to noise ratio (SNR). The method Dr. Squyres used has allowed us to store our images in order of SNR where the 17th channel image in a series has the lowest SNR and the 2nd channel image has the highest - the 1st channel image is a phase contrast image which is not used for training. 

<p align="center">
  <img alt="image" src="https://github.com/mmutadich/image-restoration-model/assets/131201068/bab38f00-274f-40dc-8e15-b73b7686c4be">
  <br /> <b> Figure 1:</b> <i>The signal-to-noise ratios (SNR’s) of the 15 images we will use to train our neural network. Using Otsu’s method on the phase contrast image of the series to calculate a threshold, we applied the threshold mask to each of the images in the rest of the series to acquire the indexes of the signal and noise pixels in each image. Using advanced indexing in Python to access the intensity values of the signal and noise pixels in each image, we could then calculate the average signal intensity and noise intensity of each image to produce a value for the SNR. This graph will allow us to compare the performance of contending neural networks at different signal-to-noise ratios which in turn allows us to more precisely select a neural network that optimally denoises between the specific range of signal-to-noise ratios that images of biofilms fall between.</i>
</p>
  
The first deep learning architecture that looked promising, because it was mentioned in multiple papers<sup>1</sup>, was a content-aware image restoration (CARE) network <sup>2</sup>. This solution has already produced consistent high-level restorations when trained on images of the flatworm Schmidtea mediterrana, which is especially sensitive to even moderate amounts of laser light so they must be photographed at low exposure times producing images of low SNR like ours. We trained the neural network on pairs of images where one has a high signal-to-noise ratio (SNR), known as the ground truth, and the other image has a low SNR. We always use the 2nd channel image in the series as our ground truth since it has the best SNR and for our initial training, we used the 10th channel image in the series as our low SNR image. 

<p align="center">
  <img alt="image" src="https://github.com/mmutadich/image-restoration-model/assets/131201068/0ca4f9a8-32e4-45a4-ba00-3941a44e57b3">
</p>

We trained the same architecture on different pairs of images where the low SNR image is different so that we can see how well the architecture performs as SNR varies. We implemented this and noticed that the model started producing unclear images after training on the 14th image of a series – this image had an SNR of 1.11. The images were unclear because although this model picked up on most bacteria in the image, it falsely identified multiple areas of the background as bacteria and in some other cases it didn’t detect some bacteria at all.

Once we independently trained multiple copies of CARE on all the channels of images in our series, we needed to quantitatively rank the performance of the neural network as SNR varies. We needed to produce a value that reflects how well the model identified the bacteria in the image - which is expressed by the number of signal pixels in each image. To do this we implemented a neural network called StarDist7 to segment our predicted images and our ground truth, returning a mask. Then we wrote an algorithm to parse through each pixel in both masks and calculate the difference between the pixels and square this value to be summed to get a final value.



# Citations
<ol>
  <li>Romain F. Laine, Guillaume Jacquemet, Alexander Krull, Imaging in focus: An introduction to denoising bioimages in the era of deep learning, The International Journal of Biochemistry & Cell Biology (2021) <a href="https://doi.org/10.1016/j.biocel.2021.106077">(https://doi.org/10.1016/j.biocel.2021.106077)</a>
  <li>Weigert, M. et al. Content-aware image restoration: pushing the limits of fuorescence microscopy (2018) <a href="https://www.nature.com/articles/s41592-018-0216-7">https://www.nature.com/articles/s41592-018-0216-7</a>
</ol>
