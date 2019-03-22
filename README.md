# Scikit Learn & Tensorflow-example-for-beginners
This is a step by step guide on implementing a scikit learn & Tensorflow-keras model. This is made for those with minimal experience in Python, some understanding of machine learning theory, and minimal experience in writing machine learning algorithems in Python.

I am making this tutorial to help non-python users in my lab group understand the very basic priniciples and algorithms behind machine learning and data analysis, however everyone is welcome to follow along and learn. 

This tutorial assumes you have Tensorflow-GPU, Python (v3.6), and iPython (Anaconda) installed on your machine.



# Downloading the files
Go to http://lesun.weebly.com/hyperspectral-data-set.html, scroll to Indian Pines, and download the following files: "corrected Indian Pines (5.7 MB)" and "Indian Pines groundtruth (1.1 KB)". Make a folder in your desktop called TF_Tutorial, and put the 2 files in there.



# Brief understanding of the data
Whenever we are working with an unfamiliar dataset, it is best to examine it before writing down any code. From the website we have we can see that we are working with <b>hyperspectral data</b> collected from an [AVRIS sensor](https://aviris.jpl.nasa.gov/), which is basically a drone that uses a [spectrometer](https://en.wikipedia.org/wiki/Spectroscopy) over a large area of land.

The first file "Indian_pines_corrected.mat" will contain a 3D array of 145 by 145 pixels, each with 224 spectral points, meaning we have a 3D matrix or array that is 145x145x224.

The second file "Indian_pines_gt.mat" will contain a 2D array of 145 by 145 pixels, with each pixel containing a value from the groundtruth table from the [website](http://lesun.weebly.com/hyperspectral-data-set.html). So for example an array containing five 4's in a row [4,4,4,4,4], will have 5 corn pixels in a row.

So just from looking at the website we can tell that the first file are our features, and the second file are the labels. This means we are able to use machine learning/deep learning/Tensorflow/Keras to classify the AVRIS sensor spectral data to crop type.


# Just what IS an array? Here's a good example!
Imagine having one of those childrens books, except this book is 145mm by 145mm and is completely blank. This book also has 224 pages. You bring a bottle of really strong black ink and spill some of it on the first page of the book. You wipe of the ink but notice that your book is covered in ink. You also notice that the ink made its way though 223 papers, and did not make it to the last page. 

Now if you think of the book as an array (145 height, 145 width, 224 pages), and the darkness of the ink ranging from values of 0 (white) to 1 (black). The first page of the ink will have 145x145 pixels that all have the value of 1.

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
    
    ```plt.plot(features[0,:,:])``` at the feature location will give us a line graph of all 224 spectral data points across the first row of pixels. This will give us a good idea of how the spectroscopy data changes as we move locations across the 145x145 map. 
    
    Open iPython again and type in ```run graphical_analysis.py``` and hit ENTER.
    
    You will see the spectroscopic image. Once you are done examining it close the window, now a line plot should appear, examine it then close it.
    
    ![](/images/imshow.png?raw=true "Title")
    ![](/images/imshow2.png?raw=true "Title")
    ![](/images/lineplot.png?raw=true "Title")
    
    Some important information we can get from the first image is that we are not working with a clean uniform image. There are also visible clusters of similarly coloured poylgons, which we can only assume to be a unique type of crop.
    
    Some important information we can get from the third image is how the Z-dimension (different spectra) change over the course of the image (please note: changes from pixel to pixel is discrete and not continuous like the line-plot implies), we are looking at only one line of pixels and not the entire image. Notice how there are some spectra lines that do not change over the course of the image? **This fact will be important later on in the machine learning process (so keep that in mind).** We can look at the spectra over the whole image, however plotting a 3D graph is not only time confusing, but can be a complete waste of time if our data is dense (which it is in our case).
    
    <b>OPTIONAL:</b> Feel free to play around with ```plt.imshow(features[:,:,0])``` by changing the value of 0 to anything from 0 to 223 in order to get a better feel of the data.
 
    <b>OPTIONAL:</b> Feel free to play around with ```plt.plot(features[0,:,:])``` by changing the value of 0 to anything from 0 to 144 in order to get a better feel of the data.
    
    
# Scikit Learn: Introduction
[Scikit-learn is a free software machine learning library for the Python programming language.](https://scikit-learn.org/stable/) Scikit learn can be a little intimidating at first, but once you have an idea of what you should be doing its fairly easy. The most difficult part about Scikit-learn is choosing the right machine learning algorithems. Ideally one should understand all the math behind all of algorithems, however if you are a beginner you should refer to the diagram below.

![](/images/ml_map.png?raw=true "Title")

Lets start at the beginning
1. At first it may seem like we have >50 samples since we have 140x140 pixels (19600 values), however we have to remember we are not working with binary data, but multi-categorial data. This means we could have 1 category 19500 values and the rest containing values smaller than 50. 
    * Open iPython, TYPE ```np.unique(labels, return_counts=True)``` and hit ENTER. You should see an array of all categories in our labels array as well as the amount of time they should up in the entire array.
    * Notice that the categories of 7 and 9 do not contain a value greater than 50.
    * Usually we would want to use a big dataset when we do any kind of machine learning, but for the sake up this example we will continue on with our analysis.
    
2. We are predicting a category (type of crop).

3. Our data is labeled (our labels array).

4. We have <100k samples.

5. It looks like we have to use [Linear SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).
    * And we will also be using [K-Nearest-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
    

# Scikit Learn: Implementation
Create a new file called ```SKL.py``` in the same directory. Copy and past the following code inside the file and save:
```python
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier

features= loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
labels= loadmat('Indian_pines_gt.mat')['indian_pines_gt']

"""NEW STUFF"""

features=features.reshape((-1,features.shape[2]))   #Flattening the 145x145 array
labels= labels.reshape(-1)

def linear_svc(n=50):
    from sklearn.svm import LinearSVC
    lin_svc=LinearSVC()
    
    from sklearn.decomposition import PCA
    pca=PCA(n_components=n)     #Default n=50
    data=pca.fit_transform(features)
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    line_svc.fit(Data_train, Labels_train)
    print (line_svc.score(Data_test, Labels_test))
    
    
def k_nn(k=19, n= 50):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)   #Default k=19
    
    from sklearn.decomposition import PCA
    pca=PCA(n_components=n)     #Default n=10
    data=pca.fit_transform(features)
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    knn.fit(Data_train, Labels_train)
    print (knn.score(Data_test, Labels_test))
```

This all looks confusing but we will go through it!


```python 
    features=features.reshape((-1,features.shape[2]))   #Flattening the 145x145 array
    labels= labels.reshape(-1) 
``` 
* Whenever you are dealing with array in machine learning, you almost always have to flatten them. In our example we turned the features 3D array to a 2D array, and the lebels 2D array to a 1D array.


```python
def linear_svc(n=50):
    from sklearn.svm import LinearSVC
    lin_svc=LinearSVC()
    
    from sklearn.decomposition import PCA
    pca=PCA(n_components=n)     #Default n=50
    data=pca.fit_transform(features)
    
    from sklearn.model_selection import train_test_split
    Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)
    
    line_svc.fit(Data_train, Labels_train)
    print (line_svc.score(Data_test, Labels_test))
```
* This is a linear svc algorithm that we talked about earlier. 
* In the first chunck of this code we import the our LinearScv model.
* In the second chunk of this code, we preform a [Principal component analysis (PCA), which is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.](https://en.wikipedia.org/wiki/Principal_component_analysis)
    * Basically PCA just checks to see if the variable we are measuring is correlated with the labels, if there are is no correlation, then it is removed from the dataset. Remember how I said earlier that there were spectra lines across our graphs that do not seem to move across our map? PCA removes those.   
* In the third chunk of this code, we use '''train_test_split''' which takes our features and labels them and splits them in to 2 parts, one part will be used to train the machine learning algorithm, the other will be used to test against to see just how good our algorithm is. Now for the 2 main parts of any machine learning algorithm, training and testing.
* In the last chunk of this code, we fit the training data to our linear_SVC model. We then measure its accuracy by using ```.score``` on the testing data.
