# Scikit Learn & Tensorflow example for beginners
This is a step by step guide on implementing a Scikit Learn & Tensorflow-keras model. This is made for those with minimal experience in Python, some understanding of machine learning theory, and minimal experience in writing machine learning algorithms in Python.

I am making this tutorial to help non-python users from various backgrounds (international and national) in my lab group understand the very basic principles and algorithms behind machine learning and data analysis, however everyone is welcome to follow along and learn. 

This tutorial assumes you have Tensorflow-GPU, Python (v3.6), and iPython (Anaconda) installed on your machine.

If at anytime in this tutorial your files stop working or you are stuck, feel free to download the python files (.py) from above. These are completed python files for the entire tutorial that will contain all the code that is used. Please do not just download and run the files from the start, try to actually follow the tutorial.

# Downloading the files
Go to http://lesun.weebly.com/hyperspectral-data-set.html, scroll to Indian Pines, and download the following files: "corrected Indian Pines (5.7 MB)" and "Indian Pines groundtruth (1.1 KB)". Make a folder in your desktop called TF_Tutorial,and put the 2 files in there.



# Brief understanding of the data
Whenever we are working with an unfamiliar dataset, it is best to examine it before writing down any code. From the website we have we can see that we are working with <b>hyperspectral data</b> collected from an [AVRIS sensor](https://aviris.jpl.nasa.gov/), which is basically a drone that uses a [spectrometer](https://en.wikipedia.org/wiki/Spectroscopy) over a large area of land.

The first file "Indian_pines_corrected.mat" will contain a 3D array of 145 by 145 pixels, each with 200 spectral points, meaning we have a 3D matrix or array that is 145x145x200.

The second file "Indian_pines_gt.mat" will contain a 2D array of 145 by 145 pixels, with each pixel containing a value from the groundtruth table from the [website](http://lesun.weebly.com/hyperspectral-data-set.html). So for example an array containing five 4's in a row [4,4,4,4,4], will have 5 corn pixels in a row.

So just from looking at the website we can tell that the first file are our features, and the second file are the labels. This means we are able to use machine learning/deep learning/Tensorflow/Keras to classify the AVRIS sensor spectral data to crop type.

So our hypothesis is: We can use spectroscopic data from an AVRIS sensor to classify different types of crops.


# Just what IS an array? Here's a good example!
Imagine having one of those children’s books, except this book is 145mm by 145mm and is completely blank. This book also has 200 pages. You bring a bottle of really strong black ink and spill some of it on the first page of the book. You wipe of the ink but notice that your book is covered in ink. You also notice that the ink made its way though 199 papers and did not make it to the last page. 

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
    ``` 
    
    Since the datafiles we got are '.mat' files we need to use ```loadmat``` to open them. Also, since 2D and 3D arrays can be hard to visualize in the mind we will use matplotlib to help visualize them. ```features``` are where the features will be stored, and ```labels``` are where the labels will be stored. 
    
    Notice how there is a weird double bracket [] after loading the file? This is because ```loadmat``` returns a dictionary with many different pieces of information, but we are only interested in the features and labels, so by using a [] we can get only the data we need.
    
    ```plt.imshow(features[:,:,0])``` at that feature location will give us idea of what image is made by one spectral data point (ex. looking at one page from our book example).
    
    ```plt.imshow(np.average(features,axis=2))``` will show us the AVRIS sensor image from the average of all the spectra. While taking the average of the spectra will never be done at any step of the machine learning process, it is cool to see what image you get.
    
    ```plt.imshow(labels)``` will show us the image of all the crops. Colours that are the same indicate the same type of crop.
    
    ```plt.plot(features[0,:,:])``` at the feature location will give us a line graph of all 200 spectral data points across the first row of pixels. This will give us a good idea of how the spectroscopy data changes as we move locations across the 145x145 map. 
    
    Open iPython again and type in ```run graphical_analysis.py``` and hit ENTER.
    
    You will see the spectroscopic image. Once you are done examining it close the window, now a line plot should appear, examine it then close it.
    
    ![](/images/imshow.png?raw=true "Single spectra slice")
    ![](/images/imshow2.png?raw=true "Average spectra")
    ![](/images/imshow3.png?raw=true "True farm area")
    ![](/images/lineplot.png?raw=true "Change in spectra over first row of pixels")
    
    Some important information we can get from the first image is that we are not working with a clean uniform image. There are also visible clusters of similarly coloured polygons, which we can only assume to be a unique type of crop.
    
    The first, second, and third image serve as a soft-visual-confirmation for our hypothesis. We can see that spectroscopic data (features) can generate an image similar to that of the actual farm land (labels).
    
    Some important information we can get from the fourth image is how the Z-dimension (different spectra) change over the course of the image (please note: changes from pixel to pixel is discrete and not continuous like the line-plot implies, also note we are looking at only one line of pixels and not the entire image). Notice how there are some spectra lines that do not change over the course of the image? **This fact will be important later on in the machine learning process (so keep that in mind).** We can look at the spectra over the whole image, however plotting a 3D graph is not only time confusing, but can be a complete waste of time if our data is dense (which it is in our case).
    
    **OPTIONAL:** Feel free to play around with ```plt.imshow(features[:,:,0])``` by changing the value of 0 to anything from 0 to 199 in order to get a better feel of the data.
 
    **OPTIONAL:** Feel free to play around with ```plt.plot(features[0,:,:])``` by changing the value of 0 to anything from 0 to 144 in order to get a better feel of the data.
    
    
# Scikit Learn: Introduction
[Scikit-learn is a free software machine learning library for the Python programming language.](https://scikit-learn.org/stable/) Scikit learn can be a little intimidating at first, but once you have an idea of what you should be doing its fairly easy. The most difficult part about Scikit-learn is choosing the right machine learning algorithms. Ideally one should understand all the math behind all of algorithms, however if you are a beginner you should refer to the diagram below.

![](/images/ml_map.png?raw=true "SKL cheat sheet")

Let’s start at the beginning
1. At first it may seem like we have >50 samples since we have 140x140 pixels (19600 values), however we have to remember we are not working with 2 categories of equal destruction, but multi-categorial data of different distribution. This means we could have 1 category with 19500 feature values and the rest containing feature values smaller than 50. 
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
1. Data preparation (This often the longest step in a real-world machine learning problem)
2. Training and testing data w/ a machine learning algorithm
3. Predicting new data


### The first algorithm we will be implementing is Linear-SVC. 
It works by generating a line that that will be used to classify the data points based on where they are relative to the line. For example, in our diagram below the Linear-SVC implementation is a bad one because both types of labels are on the same side of the line. If you were to use such implementation you would expect a score of around 0.50 (which is the worst-case scenario). The implementation on the right is the best one because each label is located on different sides of the line. This implementation would give us an accuracy of around 1. Please note that in the real world and in our example, you will almost never have an accuracy of 1. This can often be attributed to measurement error or outliers.

![](/images/Linear_SCV_guide.png?raw=true "Linear SVC guide")


### The second algorithm we will be implementing is K-nearest-neighbors.
KNN is fairly straight foreword, it classifies a new data point by looking at the 'K' closest existing labelled data points. 'K' can be set to any value of your choice. So if you choose a value of 3, the new data point will look at the nearest 3 existing labelled data points. This new datapoint will then be classified with the same label as the most common label of those 3 existing labelled data points. In the example below, the new point will be classified as blue, because of those 10 data points that are near it, the majority of them are blue.

![](/images/KNN_guide.png?raw=true "KNN example")

# Scikit Learn: RED FLAG
Did you spot anything that seemed off with what I've said so far? We were about to do a really big mistake without even noticing. I've actually made this mistake myself, and I only noticed this now, once I've finished the entire tutorial and was about to submit it to my PI. I've written all of my code for both Scikit learn and Tensorflow with this big mistake in the background. This just shows how there are always ways to make your code better and mistake free, no matter how much experience and knowledge you think you may have. 

What is the mistake? We are looking to classify *crops* using spectroscopic data. Yet half of your data is *not crops*... it's just non-crop landscape. Crops are uniform, background landscape is not. For those who are not familiar with spectroscopy, each chemical compound emits a different array of spectra, crops will have a homogenous distribution of the same compounds, while non-crops can differ drastically as they can be made from different types of organic and non-organic matter. This will mess around with the accuracy of algorithms (since the values are somewhat chaotic)! We need to remove all non-crop datapoints and labels from our dataset!

As we've seen using ```np.unique(labels, return_counts=True)```, more than half our data is non-crop which is represented by the label 0.

For those who are interested in how I figured this out, I noticed a high volatility in my algorithm scores. I thought about it and finally figured out that the only random elements from the dataset are from non-crop datapoints. 

I will highlight the fix code down below.

# Scikit Learn: Implementation
Create a new file called ```clean_data.py``` in the same directory. Copy and paste the following code inside the file and save:

```python
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
```

Create a new file called ```SKL.py``` in the same directory. Copy and past the following code inside the file and save:
```python
from clean_data import *

"""This is the linear support vector classification algorithm. We declare the algorithm, train+split, fit, predict."""
from sklearn.svm import LinearSVC
lin_svc=LinearSVC()

from sklearn.model_selection import train_test_split
Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)

lin_svc.fit(Data_train, Labels_train)
print (f'Lin_svc accuracy: {lin_svc.score(Data_test, Labels_test)}')


"""This is the k-nearest-neighbors classification algorithm. We declare the algorithm (with k=19), train+split, fit, predict."""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=19)

from sklearn.model_selection import train_test_split
Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)

knn.fit(Data_train, Labels_train)
print (f'knn accuracy: {knn.score(Data_test, Labels_test)}')
```

This all looks confusing, but we will go through it!

# Scikit Learn: Implementation Explanation
```python 
"""Flattening of the 145x145x200 features array to a 2D 21025x200 array. Flattening of 145x145 labels array to a 1D 21025 array."""
features=features.reshape((-1,features.shape[2]))   
labels= labels.reshape(-1)
``` 
* Whenever you are dealing with array in machine learning, you almost always have to flatten them. In our example we turned the features 3D array to a 2D array, and the labels 2D array to a 1D array.


```python
"""Normalizing data. This will save us a lot of processing time"""
features=normalize(features)
```
* The spectra features have values that are in the thousands so an example 3x3 spectra array would look something like this: 
    [[4000, 6000, 8000], 
    [2000, 3000, 4000], 
    [3333, 3333, 3333]]. 
* Normalization takes this array and well, normalizes the values to look something like this: 
    [[0.37139068, 0.55708601, 0.74278135], 
    [0.37139068, 0.55708601, 0.74278135], 
    [0.57735027, 0.57735027, 0.57735027]].
    * We can see [normalization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html) reduces the values of each array to be in between 0 to 1, while keeping their relative ratios (for each row of the array) to be the same.
    * This is important because when we reduce the size of numbers we can increase the speed of the algorithms.
    * This not decrease the accuracy of the algorithm because the the relative proportion is still the same between points.

```python
"""Shuffling the data and labels, while keeping their relative orders the same"""
c=list(zip(features,labels))
shuffle(c)
features,labels=zip(*c)
labels=list(labels)
```
* This is just to shuffle the data. We don't want to be getting the same results everytime we run the ML algorithms. It is important when shuffling features and labels to keep the two paired together, so that your data doesn't lose meaning (ie. each feature corresponds to the correct label despite the shuffle).
    
```python
"""PCA to reduce the number of necessary features (200 spectroscopic features). PCA is set to do it automatically."""
pca=PCA(n_components='mle', svd_solver='full')
data=pca.fit_transform(features)
print(f'Features used {len(pca.components_)}/{features.shape[1]}')
```
* [Principal component analysis (PCA), which is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.](https://en.wikipedia.org/wiki/Principal_component_analysis)
* n_components='mle', svd_solver='full' just mean the computer will just figure out which features out of the 200 are worth having for our calculations. Remember how some lines in the graph don't move throughout the figure? Those will probably be removed out.
* Most of the time when you are working with big data you want to manually choose the amount of features you have, but since our data is relatively small, we can use an automatic PCA even if it will give us some features that add little to the accuracy. In this example PCA only removes 2 spectral categories.

```python
"""Removing the non-crop data by turning all their spectra values to 0"""
for i, label in enumerate(labels):
    if label==0:
        data[i]=np.zeros((data.shape[1],))
```
* This code is pretty self explanatory it just reduces all our background non-crop data to 0, this is important because there is not way to classify something as background to the choatic nature of the background.

```python
"""This is the linear support vector classification algorithm. We declare the algorithm, train+split, fit, predict."""
from sklearn.svm import LinearSVC
lin_svc=LinearSVC()

from sklearn.model_selection import train_test_split
Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)

lin_svc.fit(Data_train, Labels_train)
print (f'Lin_svc accuracy: {lin_svc.score(Data_test, Labels_test)}')
```
* This is a linear svc algorithm that we talked about earlier. 
* In the first chunk of this code we import the our LinearSvc model. 
* In the second chunk of this code, we use ```train_test_split``` which takes our features and labels them and splits them in to 2 parts, one part will be used to train the machine learning algorithm, the other will be used to test against to see just how good our algorithm is. Now for the 2 main parts of any machine learning algorithm, training and testing.
* In the third chunk of this code, we fit the training data to our linear_SVC model. We then measure its accuracy by using ```.score``` on the testing data.


```python
"""This is the k-nearest-neighbors classification algorithm. We declare the algorithm (with k=19), train+split, fit, predict."""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=19)

from sklearn.model_selection import train_test_split
Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)

knn.fit(Data_train, Labels_train)
print (f'knn accuracy: {knn.score(Data_test, Labels_test)}')
```
* This is the exact same implementation as linear_SVC, but instead we are using k-nearest-neighbors and sitting the value of k to 19.

Please note that the original spectra data had 224 features (different spectra) but we downloaded the corrected file which reduced the number of features to 200, which is why the PCA won't remove much features in this example.


# Scikit Learn: Testing
* Go back to iPython, TYPE ```run SKL.py``` and hit ENTER.
* After 5 more seconds the accuracy for lin_svc should pop up.
    * It should be around 0.68
* After 15 more seconds the accuracy for knn should pop up.
    * It should be around 0.88
* Which algorithm gives the best accuracy?
* See what happens when you run the code again (TYPE ```run SKL.py``` and hit ENTER)
    * Got different accuracy values? This is because we are training and testing on different data every time we run the code(test_size=0.33)
* The knn algorithm is clearly superior for this example (when it comes to accuracy) with an average accuracy of around 0.88, compared to linear_SVC which gives an average accuracy of 0.68.


# Scikit Learn: Interpreting our results (Optional)
Is an accuracy value of 0.88 good? This is an overall accuracy value so it might not tell us the complete story. Since we have 17 categories we could get an accuracy value for one type of crop to be 1.0, while another type of crop at 0.5.

Normally we would check our model against new data, but unfortunately we do not have any. So the next best thing is to check it against the test_data that we got from splitting.

Copy and paste the following code into iPython or put it in the file (then rerun the file).
```python
"""Visualization of results"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

array=normalize(confusion_matrix(Labels_test, knn.predict(Data_test)))
  
df_cm = pd.DataFrame(array, range(17),
                  range(17))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Greens')# font size
plt.xlabel('Predicted label', fontsize=16)
plt.ylabel('True label', fontsize=16)
plt.show()
```
This will give us a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) that should something like this.
![](/images/SKL_prediction2.png?raw=true "KNN Confusion Matrix")
Confusion matrices are useful for showing us the strengths and the weaknesses of our multi-categorical machine learning algorithm. True positives (correct predictions) are placed along the diagonal of the chart.  
As we can see we have good accuracy for most types of crops, but there are some crops with low accuracy values and some with values of 0. This can be attributed to the following reasons
* Small training size and/or size imbalances between categories (which I believe is the primary culprit)
    * Just type ```np.unique(labels, return_counts=True)``` and you will see that labels with low sample number have the lowest accuracy values in the confusion matrix
* Similar categories
* A truly random category
* Algorithm just does not work for this type of category

Just for fun, here is a comparison between the true farm data against the predicted farm data.
![](/images/imshow3.png?raw=true "True map")
![](/images/SKL_prediction.png?raw=true "Predicted map")


# Scikit Learn: Optimization
We now have our models set and ready to use, but is there a way to make them even better? We can try to optimize them by adding in parameters when we call our algorithm method. Try playing around with the value of k in k_nn and see what you get. Optimization too much of a complicated topic for this tutorial so we are going to end it there with Scikit-Learn. I will make a whole separate tutorial on both Scikit learn and Tensorflow optimization later on. However if you would like to try to play around with the idea of optimization read the documentation of [linear SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) and [knn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).


# Tensorflow: Theory
Tensorflow uses neural networks which are a type of machine learning algorithm. I could spend a while explaining all of the theory behind how the algorithm works, but I think Youtube does a much better job. Here is a [short and brief video](https://www.youtube.com/watch?v=rEDzUT3ymw4) and a [longer and detailed video](https://www.youtube.com/watch?v=aircAruvnKk) video explaining the Tensorflow algorithm. Of the two, I recommend watching the longer video.

In my opinion a reason to use Tensorflow against regular machine learning algorithms is that (generally speaking) neural networks take into account relationships between different features, while most machine learning algorithms just use the relationship between the features and the labels.

Reasons to not to use neural networks (NN) in comparison to regular machine learning algorithms (MLA) include:
* Harder to implement
* Harder to optimize
* Longer time to execute
* Sometimes regular MLA outperform NN


# Tensorflow: Implementation
Create a new file called ```tf_tut.py``` in the same directory. Copy and paste the following code inside the file and save:
```python
from clean_data import *
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

"""Tensorflow model code"""
model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dropout(0.5))


model.add(tf.keras.layers.Dense(17, activation=tf.nn.softmax))  # our output layer. 17 units for 17 classes. Softmax for probability distribution


model.compile(optimizer=Adam(lr=0.0015),  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(data, labels,
                  epochs=100,
                  batch_size=32,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5),])  # train the model
```
Please note I reused a lot of code from a Youtuber named "Sentdex" and fit it to this problem, so if you want to know more about him click [here](https://www.youtube.com/user/sentdex). 

# Tensorflow: Implementation Explanation
```python
model = tf.keras.models.Sequential()  # a basic feed-forward model
```
* *If you have not watch the videos I linked to above, you will have a hard time understanding what is going on here.*
* All NNs use models. [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential) is a type of model that just makes a linear stack of the models that we will call after this line of code. Sequential models do not share layers or have multiple inputs or outputs.
* The other type of Tensorflow model which we will not be using is a [functional](https://www.tensorflow.org/alpha/guide/keras/functional) model, which allows for different layers to connect to each other, as well as multiple inputs and outputs.


```python
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
```
* A dense layer is just a layer of neurons that [receives input from the neurons which are before it](https://cdn-images-1.medium.com/max/1200/1*eJ36Jpf-DE9q5nKk67xT0Q.jpeg). The first dense layer here will receive input from the features we made earlier and output it to the next. The second layer will receive input from the 1st layer and do the same, and so and and so forth. The reason why there are 5 layers here instead of 1 is because we have many complex features (200 spectra) that all have unique and complicated interactions with each other. Generally speaking, the more complicated the interaction between features the more layers you will have. 
* 128 neurons were chosen for this example just because they gave me the best results from the start, I'm sure a different number will be chosen once I have optimized this model.
* An activation function of ‘relu’ stands for rectified linear unit. It kind of looks [like this] (https://cdn-images-1.medium.com/max/1200/1*oePAhrm74RNnNEolprmTaQ.png). Generally speaking this is how it works, if the weighted sum of the input + the bias is less than 0, the neuron will not activate because its value will equal to 0 (this assumes that this neuron is only connected to one other neuron). However if weighted sum of the input + the bias is less than 0, it will cause the neuron to activate because its value will be equal the value of the weighted sum of the input + the bias (this assumes that this neuron is only connected to one other neuron). You can read more about activation functions [here](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0). Relu is usually the go-to activation function as it preforms well and is quick.

```python
model.add(tf.keras.layers.Dropout(0.5))
```
* This is a dropout layer. The basic idea behind a dropout layer is to remove a set amount of neurons from the NN algorithm during every epoch. It is typically used when a NN is has [overfiting](https://en.wikipedia.org/wiki/Overfitting) issues. After each run 0.5 or 50% of the neurons will be *randomly* removed from the NN. In this example it is not necessary to use one, however I noticed some slight overfitting problems. I will further explain when I talk about optimizing the model.

```python
model.add(tf.keras.layers.Dense(17, activation=tf.nn.softmax))  # our output layer. 17 units for 17 classes. Softmax for probability distribution
```
* This looks similar to the layers above, but there is a difference. The last layer of any sequential netowrk will be the otuput layer. The value of 17 indicates how many categories we have, which is 17 in our case. We are not using relu but [softmax](https://www.researchgate.net/profile/Binghui_Chen/publication/319121953/figure/fig2/AS:527474636398592@1502771161390/Softmax-activation-function.png).


```python
model.compile(optimizer=Adam(lr=0.0015),  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track
```
* Now that we have made all of the necessary layers of our model, we have to compile the model. 
* The model includes an optimizer which is set to 'Adam'. Remember in the video you watched earlier how it said that every time the algorithm goes through the NN, the algorithm will go back and adjusts the weights of the connections between neurons so that we can minimize loss? That's what the optimizer does. 'Adam' is just the go-to optimizer in most basic NN, however if you want to explore other optimizers [click here](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer).
    * lr is the learning rate. Change the values to change how fast the NN learns.
* Loss measures how 'good' our model is. It is what we are trying to minimize. 'Sparse_categorical_crossentropy' refers to the type of loss function we will be using. There are other functions you can explore by [clicking here](https://www.tensorflow.org/api_docs/python/tf/losses).
* Metrics just refers to what we want to track as the model is running. In this case we are monitoring accuracy. If you wish to track more things [click here](https://www.tensorflow.org/api_docs/python/tf/metrics).

```python
model.fit(data, labels,
                  epochs=100,
                  batch_size=32,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(patience=5),])  # train the model
```
* Now we are training the model on our data.
* 'Epochs' refers to the number of times we want our NN to go through the data and optimize. In this case it will do so 100 times.
* 'Batch_size' refers to the number of data points that will be fed into the NN algorithm at a time. In this case 32 data points are put in at a time.
* 'Validation_split' refers to the percentage of the training data that will be placed aside to be tested on. This is equivalent to the scikit learn function ```Data_train, Data_test, Labels_train, Labels_test = train_test_split(data, labels, test_size=0.33)```. In this case we will leave 10% of the data to test on.
* 'Callbacks' are just functions that will be excuted during the training process. In this case we are running the function 'EarlyStopping' with a patience value of 5. This fuction will stop the NN from running once it notices that training does not improve the function. It will tolerate only 5 epochs of where the NN does not improve (in terms of loss). There are more callback functions that you can read about by [clicking here](https://keras.io/callbacks/). 

# Tensorflow: Note of caution
The code that I have made is very basic and has obvious flaws. It was written just so you can understanding the basics behind the structure of a neural network. A better version of the code will be made during my optimization tutorial.


# Tensorflow Learn: Testing
* Go back to iPython, TYPE ```run tf_tut.py``` and hit ENTER.
* You should see something that looks like this ![](/images/tf2.png?raw=true "TF Results")
* Here we can see the training loss, training accuracy, validation loss, and validation accuracy for each epoch
* Depending on the validation split, you will see different results, your epochs might stop at 8, and you might get different accuracy values.
* In my case, Tensorflow gave me better values than Scikit learn, this might be different for you.
* You will notice that if you keep running the model, you will keep seeing different results, once again this is because of a low number of data and their uneven distribution between different categories.
* Please note, that when you want to run the TF model I recommend you do not type ```run tf_tut.py``` and hitting enter again. Instead type ```exit()``` hit enter, then type ```ipython``` hit enter, then type ```run tf_tut.py``` and hit enter. This is just so you can  get a new randomized dataset everytime. Also if you are on a weak computer, opening and closing iPython will help to clean your RAM.

# Tensorflow Learn: Interpreting our results (Optional)
Is an validation accuracy value of 0.95 good? This is an overall accuracy value so it might not tell us the complete story. Since we have 17 categories we could get an accuracy value for one type of crop to be 1.0, while another type of crop at 0.5.

Normally we would check our model against new data, but unfortunately we do not have any. So the next best thing is to check it against the test_data that we got from splitting.

Copy and paste the following code into iPython or put it in the file (then rerun the file).
```python
"""Visualization of results"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

array=normalize(confusion_matrix(model.history.validation_data[1], model.predict_classes(model.history.validation_data[0])))
  
df_cm = pd.DataFrame(array, range(17),
                  range(17))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Blues')# font size
plt.xlabel('Predicted label', fontsize=16)
plt.ylabel('True label', fontsize=16)
plt.show()
```
This will give us a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) that should something like this.
![](/images/TF_prediction2.png?raw=true "TF Confusion Matrix")
Notice how these values are much better than the KNN confusion matrix? This is the power of NN. And this is with just some optimization being implemented (which I casually inserted into the code).
Please note if you get a row with only 0's in the heat map, just reload ```clean_data.py``` in order to get newly shuffled data. It could just be that all the data from one category ended up in the validation set (by pure chance).

Just for fun, here is a comparison between the true farm data against the KNN predicted farm data and the TF predicted farm data (in that order).
![](/images/imshow3.png?raw=true "True map")
![](/images/SKL_prediction.png?raw=true "KNN Predicted map")
![](/images/TF_prediction.png?raw=true "TF Predicted map")
Again look how close the TF map is to the true map, compared with the KNN map to the true map.


# TF Learn: Optimization
TF optimization is hard, and takes a lot of time to implement. Which is why it will be left for another day. Even the worlds top machine learning experts and TF users have a hard time implementing the optimum TF parameters. Also remember that there is no 'correct' algorithm optimization. A lot of TF optimization involves trial and error.
To get better at using TF I recommend to keep using TF on other problems as well as familiarizing yourself with the theory behind the algorithm (statistics, calculus, probability).

# Closing Remarks
