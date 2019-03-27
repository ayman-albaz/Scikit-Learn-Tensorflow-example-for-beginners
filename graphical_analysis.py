from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

"""Loading our features and labels from the matlab file"""
features= loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
labels= loadmat('Indian_pines_gt.mat')['indian_pines_gt']

"""Graphical analysis"""
plt.imshow(features[:,:,0]) #145x145 image of a single spectra
plt.show()
plt.imshow(np.average(features,axis=2)) #145x145 image of the average of all (200) spectra
plt.show()
plt.imshow(labels) #Heatmap of different crops
plt.show()
plt.plot(features[0,:,:]) #200 spectra distribution along the first 145 pixels
plt.show()

