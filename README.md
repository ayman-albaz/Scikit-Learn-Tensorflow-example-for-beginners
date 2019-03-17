# Tensorflow-example-for-beginners
This is a step by step guide on implementing a Tensorflow-keras model. This is made for those with minimal experience in Python, some understanding of machine learning theory, and minimal experience in writing machine learning algorithems in Python.

I am making this tutorial to help non-python users in my lab group understand the very basic priniciples and algorithms behind machine learning and data analysis, however everyone is welcome to follow along and learn. 

This tutorial assumes you have Tensorflow-GPU, Python (v3.6), and iPython installed on your machine.



# Downloading the files
Go to http://lesun.weebly.com/hyperspectral-data-set.html, scroll to Indian Pines, and download the following files: "corrected Indian Pines (5.7 MB)" and "Indian Pines groundtruth (1.1 KB)". Make a folder in your desktop called TF_Tutorial, and put the 2 files in there.



# Brief understanding of the data
Whenever we are working with an unfamiliar dataset, it is best to examine it before writing down any code. From the website we have we can see that we are working with <b>hyperspectral data</b> collected from an [AVRIS sensor](https://aviris.jpl.nasa.gov/), which is basically a drone that uses [spectrometer](https://en.wikipedia.org/wiki/Spectroscopy) over a large area of land.

The first file "Indian_pines_corrected.mat" will contain a 3D array of 145 by 145 pixels, each with 224 spectral points, meaning we have a 3D matrix or array that is 145x145x224.

The second file "Indian_pines_gt.mat" will contain a 2D array of 145 by 145 pixels, with each pixel containing a value from the groundtruth table from the [website](http://lesun.weebly.com/hyperspectral-data-set.html). So for example an array containing five 4's in a row [4,4,4,4,4], will have 5 corn pixels in a row.

So just from looking at the website we can tell that the first file are our features, and the second file are the labels. This means we are able to use machine learning/deep learning/Tensorflow/Keras to classify the AVRIS sensor spectral data to crop type.


# Just what IS an array? Here's a good example!
Imagine having one of those childrens books, except this book is 145mm by 145mm and is completely blank. This book also has 224 pages. You bring a bottle of really strong ink and spill some of it on the first page of the book. You wipe of the ink but notice that your book is covered in ink. You also notice that the ink made its way though 223 papers, and did not make it to the last page. 

Now if you think of the book as an array (145 height, 145 width, 224 pages), and the darkness of thean ink ranging from values of 1 to 0. The first page of the ink will have 145x145 pixels that all have the value of 1.

As you keep turning the pages you notice the values of the 145x145 pixels decreasing, making their way from 1 to 0.

The last page will have all 145x145 pixels with a value of 0, as the ink did not reach that page.



# Deeper understanding of the data
1. Open the folder "TF_Tutorial", Create a file in that folder called "tutorial.py". Hold the SHIFT key and RIGHT CLICK anywhere in the white space of the folder. LEFT CLICK on "Open PowerShell window here". TYPE ```ipython``` and hit ENTER. You should now have ipython open.

2. Minimize iPython and edit the "tutorial.py" file using notepad, notepad++ or your favourite IDE.

3. Type in the following in "tutorial.py":
    ```python
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    features= loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    labels= loadmat('Indian_pines_gt.mat')['indian_pines_gt']

    def data_plots():
        plt.imshow(features[:,:,0])
        plt.show()
        plt.plot(features[0,:,:])
        plt.show()
    
    ``` 
    Since the datafiles we got are '.mat' files we need to use loadmat to open them. Also since 2D and 3D arrays can be hard to visualize in the mind we will use matplotlib to help visualize them. <b>features</b> are where the features will be stored, and <b>labels</b> are where the labels will be stored. 
    
    Notice how there is a weird double bracket [] after loading the file? This is because loadmat returns a dictionary with many different pieces of information, but we are only interested in the features and labels, so by using a [] we can get only the data we need.
    
    ```plt.imshow``` at that feature location will give us idea of what image is made by one spectral data point (ie. looking at one page from out book example). 
    
    ```plt.plot``` at the feature location will give us a line graph of all 224 spectral data points across the first row of pixels. This will give us a good idea of how the spectroscopy data changes as we move locations across the 145x145 map. 
    
    Open iPython again and type in ```run tutorial.py``` and hit ENTER.
    
    TYPE in ```data_plots()``` and hit ENTER. You will see the spectroscopic image. Once you are done examining it close the window, now a line plot should appear, examine it then close it.
    
    <b>OPTIONAL:</b> Feel free to play around with ```plt.imshow(features[:,:,0])``` by changing the value of 0 to anything from 0 to 223 in order to get a better feel of the data.
 
    <b>OPTIONAL:</b> Feel free to play around with ```plt.plot(features[0,:,:])``` by changing the value of 0 to anything from 0 to 144 in order to get a better feel of the data.
