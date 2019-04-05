from scipy.io import loadmat
import numpy as np
from random import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

"""Loading our features and labels from the matlab file"""
features= loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
labels= loadmat('Indian_pines_gt.mat')['indian_pines_gt']

"""Flattening of the 145x145x200 features array to a 2D 21025x200 array. Flattening of 145x145 labels array to a 1D 21025 array."""
features=features.reshape((-1,features.shape[2]))   
labels= labels.reshape(-1)

"""Shuffling the data and labels, while keeping their relative orders the same"""
c=list(zip(features,labels))
shuffle(c)
features,labels=zip(*c)
labels=np.array(labels)

"""Normalizing data. This will save us alot of processing time"""
features=normalize(features)

"""PCA to reduce the amount of necessary features (200 spectroscopic features). PCA is set to do it automatically."""
pca=PCA(n_components='mle', svd_solver='full')
data=pca.fit_transform(features)
print(f'Features used {len(pca.components_)}/{features.shape[1]}')


"""Removing the non-crop data by turning all their spectra values to 0"""
for i, label in enumerate(labels):
    if label==0:
        data[i]=np.zeros((data.shape[1],))
        
        
