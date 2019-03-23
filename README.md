# Scikit Learn & Tensorflow-example-for-beginners
This is a step by step guide on implementing a scikit learn & Tensorflow-keras model. This is made for those with minimal experience in Python, some understanding of machine learning theory, and minimal experience in writing machine learning algorithems in Python.

I am making this tutorial to help non-python users in my lab group understand the very basic priniciples and algorithms behind machine learning and data analysis, however everyone is welcome to follow along and learn. 

This tutorial assumes you have Tensorflow-GPU, Python (v3.6), and iPython (Anaconda) installed on your machine.



# Downloading the files
Go to http://lesun.weebly.com/hyperspectral-data-set.html, scroll to Indian Pines, and download the following files: "corrected Indian Pines (5.7 MB)" and "Indian Pines groundtruth (1.1 KB)". Make a folder in your desktop called TF_Tutorial, and put the 2 files in there.



# Brief understanding of the data
Whenever we are working with an unfamiliar dataset, it is best to examine it before writing down any code. From the website we have we can see that we are working with <b>hyperspectral data</b> collected from an [AVRIS sensor](https://aviris.jpl.nasa.gov/), which is basically a drone that uses a [spectrometer](https://en.wikipedia.org/wiki/Spectroscopy) over a large area of land.

The first file "Indian_pines_corrected.mat" will contain a 3D array of 145 by 145 pixels, each with 200 spectral points, meaning we have a 3D matrix or array that is 145x145x200.

The second file "Indian_pines_gt.mat" will contain a 2D array of 145 by 145 pixels, with each pixel containing a value from the groundtruth table from the [website](http://lesun.weebly.com/hyperspectral-data-set.html). So for example an array containing five 4's in a row [4,4,4,4,4], will have 5 corn pixels in a row.

So just from looking at the website we can tell that the first file are our features, and the second file are the labels. This means we are able to use machine learning/deep learning/Tensorflow/Keras to classify the AVRIS sensor spectral data to crop type.


# Just what IS an array? Here's a good example!
Imagine having one of those childrens books, except this book is 145mm by 145mm and is completely blank. This book also has 200 pages. You bring a bottle of really strong black ink and spill some of it on the first page of the book. You wipe of the ink but notice that your book is covered in ink. You also notice that the ink made its way though 199 papers, and did not make it to the last page. 

Now if you think of the book as an array (145 height, 145 width, 200 pages), and the darkness of the ink ranging from values of 0 (white) to 1 (black). The first page of the ink will have 145x145 pixels that all have the value of 1.

As you keep turning the pages you notice the values of the 145x145 pixels decreasing, making their way from 1 to 0.

The last page will have all 145x145 pixels with a value of 0, as the ink did not reach that page.



# Deeper understanding of the data
1. Open the folder "TF_Tutorial", Create a file in that folder called "graphical_analysis.py". Hold the SHIFT key and RIGHT CLICK anywhere in the white space of the folder. LEFT CLICK on "Open PowerShell window here". TYPE ```ipython``` and hit ENTER. You should now have ipython open.

2. Minimize iPython and edit the "graphical_analysis.py" file using notepad, notepad++ or your favourite IDE.

3. Type in the following in "graphical_analysis":
    ```python
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import numpy as np

    features= loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    labels= loadmat('Indian_pines_gt.mat')['indian_pines_gt']

    plt.imshow(features[:,:,0])
    plt.show()
    plt.imshow(np.average(features,axis=2))
    plt.show()
    plt.plot(features[0,:,:])
    plt.show()
    
    ``` 
    Since the datafiles we got are '.mat' files we need to use loadmat to open them. Also since 2D and 3D arrays can be hard to visualize in the mind we will use matplotlib to help visualize them. ```features``` are where the features will be stored, and ```labels``` are where the labels will be stored. 
    
    Notice how there is a weird double bracket [] after loading the file? This is because loadmat returns a dictionary with many different pieces of information, but we are only interested in the features and labels, so by using a [] we can get only the data we need.
    
    ```plt.imshow(features[:,:,0])``` at that feature location will give us idea of what image is made by one spectral data point (ex. looking at one page from our book example).
    
    ```plt.imshow(np.average(features,axis=2))``` will show us the AVRIS sensor image from the average of all the spectra. While taking the average of the spectra will never be done at any step of the machine learning process, it is cool to see what image you get.
    
    ```plt.plot(features[0,:,:])``` at the feature location will give us a line graph of all 200 spectral data points across the first row of pixels. This will give us a good idea of how the spectroscopy data changes as we move locations across the 145x145 map. 
    
    Open iPython again and type in ```run graphical_analysis.py``` and hit ENTER.
    
    You will see the spectroscopic image. Once you are done examining it close the window, now a line plot should appear, examine it then close it.
    
    ![](/images/imshow.png?raw=true "Title")
    ![](/images/imshow2.png?raw=true "Title")
    ![](/images/lineplot.png?raw=true "Title")
    
    Some important information we can get from the first image is that we are not working with a clean uniform image. There are also visible clusters of similarly coloured poylgons, which we can only assume to be a unique type of crop.
    
    Some important information we can get from the third image is how the Z-dimension (different spectra) change over the course of the image (please note: changes from pixel to pixel is discrete and not continuous like the line-plot implies, also note we are looking at only one line of pixels and not the entire image). Notice how there are some spectra lines that do not change over the course of the image? **This fact will be important later on in the machine learning process (so keep that in mind).** We can look at the spectra over the whole image, however plotting a 3D graph is not only time confusing, but can be a complete waste of time if our data is dense (which it is in our case).
    
    <b>OPTIONAL:</b> Feel free to play around with ```plt.imshow(features[:,:,0])``` by changing the value of 0 to anything from 0 to 199 in order to get a better feel of the data.
 
    <b>OPTIONAL:</b> Feel free to play around with ```plt.plot(features[0,:,:])``` by changing the value of 0 to anything from 0 to 144 in order to get a better feel of the data.
    
    
# Scikit Learn: Introduction
[Scikit-learn is a free software machine learning library for the Python programming language.](https://scikit-learn.org/stable/) Scikit learn can be a little intimidating at first, but once you have an idea of what you should be doing its fairly easy. The most difficult part about Scikit-learn is choosing the right machine learning algorithms. Ideally one should understand all the math behind all of algorithms, however if you are a beginner you should refer to the diagram below.

![](/images/ml_map.png?raw=true "Title")

Lets start at the beginning
1. At first it may seem like we have >50 samples since we have 140x140 pixels (19600 values), however we have to remember we are not working with binary data, but multi-categorial data. This means we could have 1 category 19500 values and the rest containing values smaller than 50. 
    * Open iPython, TYPE ```np.unique(labels, return_counts=True)``` and hit ENTER. You should see an array of all categories in our labels array as well as the amount of time they should up in the entire array.
    * Notice that the categories of 7 and 9 do not contain a value greater than 50.
    * Usually we would want to use a big dataset when we do any kind of machine learning, but for the sake up this example we will continue on with our analysis.
    
2. We are predicting a category (type of crop).

3. Our data is labeled (our labels array).

4. We have <100k samples.

5. It looks like we have to use [Linear Support Vector Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
    * And we will also be using [K-Nearest-Neighbors (KNN)](https://scikit-learn.org/stable/modules/neighbors.html)


# Scikit Learn: Theory
Any machine learning implementation will involve the following steps.
1. Data preperation (This often the longest step in a real-world machine learning problem)
2. Training and testing data w/ a machine learning algorithm
3. Predicting new data


### The first algorithm we will be implementing is Linear-SVC. 
It works by generating a line that that will be used to classify the data points based on where they are relative to the line. For example, in our diagram below the Linear-SVC implementation is a bad one because both types of labels are on the same side of the line. If you were to use such implementation you would expect a score of around 0.50 (which is the worst case scenario). The implementation on the right is the best one because each label is located on different sides of the line. This implementation would give us an accuracy of around 1. Please note that in the real world and in our example, you will almost never have an accuracy of 1. This can often be attributed to measurement error or outliers.

![](/images/Linear_SCV_guide.png?raw=true "Title")


### The second algorithm we will be implementing is K-nearest-neighbors.
KNN is fairly straight foreward, it classifies a new data point by looking at the 'K' closest existing labelled data points. 'K' can be set to any value of your choice. So if you choose a value of 3, the new data point will look at the nearest 3 existing labelled data points. This new datapoint will then be classified with the same label as the most common label of those 3 existing labelled data points. In the example below, the new point will be classified as blue, because of those 10 data points that are near it, the majority of them are blue.

![](/images/KNN_guide.png?raw=true "Title")

# Scikit Learn: Implementation
Create a new file called ```SKL.py``` in the same directory. Copy and past the following code inside the file and save:
```python
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
    print (lin_svc.score(Data_test, Labels_test))
    
    
def k_nn():
    """This is the k-nearest-neighbors classification algorithm. We declare the algorithm (with k=19), train+split, fit, predict."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=19)
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    knn.fit(Data_train, Labels_train)
    print (f'Accuracy: {knn.score(Data_test, Labels_test)}')
```

This all looks confusing but we will go through it!

# Scikit Learn: Implementation Explanation
```python 
"""Flattening of the 145x145x200 features array to a 2D 21025x200 array. Flattening of 145x145 labels array to a 1D 21025 array."""
features=features.reshape((-1,features.shape[2]))   
labels= labels.reshape(-1)
``` 
* Whenever you are dealing with array in machine learning, you almost always have to flatten them. In our example we turned the features 3D array to a 2D array, and the lebels 2D array to a 1D array.


```python
"""Normalizing data. This will save us alot of processing time"""
features=normalize(features)
```
* The spectra features have values that are in the thousands so an example 3x3 spectra array would look something like this: [[4000, 6000, 8000], [2000, 3000, 4000], [3333, 3333, 3333]]. Normalization takes this array and well, normalizes the values to look something like this: [[0.37139068, 0.55708601, 0.74278135], [0.37139068, 0.55708601, 0.74278135], [0.57735027, 0.57735027, 0.57735027]]
    * We can see normalization reduces the values of each array to be in between 0 to 1, while keeping their relative proportions (for each row of the array) to be the same.
    * This is important because when we reduce the size of numbers we can increase the speed of the algorithms.
    * This not decrease the accuracy of the algorithm because the the relative proportion is still the same between points.
    
    
```python
"""PCA to reduce the amount of necessary features (200 spectroscopic features). PCA is set to do it automatically."""
pca=PCA(n_components='mle', svd_solver='full')
data=pca.fit_transform(features)
print(f'Features used {len(pca.components_)}/{features.shape[1]}')
```
* [Principal component analysis (PCA), which is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.](https://en.wikipedia.org/wiki/Principal_component_analysis)
* n_components='mle', svd_solver='full' just mean the computer will just figure out which features out of the 200 are worth having for our calculations. Remember how some lines in the graph don't move throughout the figure? Those will probably be removed out.
* Most of the time when you are working with big data you want to manually choose the amount of features you have, but since our data is relatively small, we can use an automatic PCA even if it will give us some features that add little to the accuracy.


```python
def linear_svc():
    """This is the linear support vector classification algorithm. We declare the algorithm, train+split, fit, predict."""
    from sklearn.svm import LinearSVC
    lin_svc=LinearSVC()
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    lin_svc.fit(Data_train, Labels_train)
    print (f'Accuracy: {lin_svc.score(Data_test, Labels_test)}')
```
* This is a linear svc algorithm that we talked about earlier. 
* In the first chunck of this code we import the our LinearScv model. 
* In the second chunk of this code, we use ```train_test_split``` which takes our features and labels them and splits them in to 2 parts, one part will be used to train the machine learning algorithm, the other will be used to test against to see just how good our algorithm is. Now for the 2 main parts of any machine learning algorithm, training and testing.
* In the third chunk of this code, we fit the training data to our linear_SVC model. We then measure its accuracy by using ```.score``` on the testing data.


```python
def k_nn():
    """This is the k-nearest-neighbors classification algorithm. We declare the algorithm (with k=19), train+split, fit, predict."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=19)
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    knn.fit(Data_train, Labels_train)
    print (f'Accuracy: {knn.score(Data_test, Labels_test)}')
```
This is the exact same implementation as linear_SVC, but instead we are using k-nearest-neighbors and sitting the value of k to 19.
