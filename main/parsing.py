''''''
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

'''
    Parses through the phase contrast image
    Applies ostu's threshold to return a mask
    Parses each image and and calculates SNR 
    Adds key: exposure time, value: SNR to dictionary

    Arguments:
        - times (list of strings): The exposure times that will be added as keys in the dictionary

    Returns:
        - data (dictionary): Dictionary containing keys of exposure times and values of SNR
'''
def SNR_exposure_time(times):
    data = {}
    with open('Planes/file_1.tif', 'rb') as file:
            img = pickle.load(file)  
            img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            mask = otsu(img)
    for i in range(1, 16):
        with open('Planes/file_'+str(i+1)+'.tif', 'rb') as file:
            img = pickle.load(file)  
            img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            data[times[i]] = SNR(img, mask) #add ratio to dictionary
    return data
    
    
'''
    Given the image and the mask, uses advanced indexing to find the pixel intensities
    at the indexes that are 'signal' and 'noise
    Find the mean of the signal and noise intensities and calculates a ratio

    Arguments:
        - img (np.array): The image to caluclate the SNR 
        - mask (np.array): The mask to be used to binarise the image and calculate SNR

    Returns:
        - ratio (float): SNR of image
'''
def SNR(img, mask):
    noise = img[~mask]
    signal = img[mask]
    average_signal = np.mean(signal)
    average_noise = np.mean(noise)
    ratio = average_signal/average_noise
    print(ratio)
    return ratio

'''
    Plots histogram of the pixel intensities of an image

    Arguments:
        - img (np.array): The image to display the intensities
'''
def distribution(img):
    # find frequency of pixels in range 0-255
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.show()
    SNR(img, 103)
    plt.show()

'''
    Given an image, calculates otsu's threshold to get the threshold and the mask

    Arguments:
        - img (np.array): The image to caluclate otsu's threshold

    Returns:
        - thresh1 (np.array): mask calulated by otsu's threshold
'''
def otsu (img):
    ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)
    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 1024, 1022)
    cv2.imshow("Resized_Window", thresh1) #signal is a 0, noise is a 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(thresh1)

'''
    Given an image, displays it

    Arguments:
        - img (np.array): The image to display

'''
def display (image_0):
    plt.imshow(image_0, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    exposure_times = ['10s', '8s', '6s', '4s', '2s', '1s',
             '800 ms', '600 ms', '400ms', '200ms', '100ms',
               '80ms', '60ms', '40ms', '20ms', '20ms']
    dict = SNR_exposure_time(exposure_times)
    courses = list(dict.keys())
    print(courses)
    values = list(dict.values())
    print(values)
    fig = plt.figure(figsize = (10, 5))
    # creating the bar plot
    plt.bar(courses, values, color ='maroon',
            width = 1)
    plt.show()
