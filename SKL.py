from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


"""Loading our features and labels from the matlab file"""
features= loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
labels= loadmat('Indian_pines_gt.mat')['indian_pines_gt']

"""Flattening of the 145x145x200 features array to a 2D 21025x200 array. Flattening of 145x145 labels array to a 1D 21025 array."""
features=features.reshape((-1,features.shape[2]))   
labels= labels.reshape(-1)

"""Normalizing data. This will save us alot of processing time"""
features=normalize(features)

"""PCA to reduce the amount of necessary features (200 spectroscopic features). PCA is set to do it automatically."""
pca=PCA(n_components='mle', svd_solver='full')
data=pca.fit_transform(features)
print(f'Features used {len(pca.components_)}/{features.shape[1]}')


def linear_svc():
    """This is the linear support vector classification algorithm. We declare the algorithm, train+split, fit, predict."""
    from sklearn.svm import LinearSVC
    lin_svc=LinearSVC()
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    lin_svc.fit(Data_train, Labels_train)
    print (f'Accuracy: {lin_svc.score(Data_test, Labels_test)}')
    
    
def k_nn():
    """This is the k-nearest-neighbors classification algorithm. We declare the algorithm (with k=19), train+split, fit, predict."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=19)
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    knn.fit(Data_train, Labels_train)
    print (f'Accuracy: {knn.score(Data_test, Labels_test)}')
    
    