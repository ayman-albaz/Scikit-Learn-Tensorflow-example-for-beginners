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


                  
"""Visualization of results"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

array=normalize(confusion_matrix(model.history.validation_data[1], model.predict_classes(model.history.validation_data[0])))
  
df_cm = pd.DataFrame(array, range(17),
                  range(17))
sn.set(font_scale=1.4)  #for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10}, cmap='Blues')# font size
plt.xlabel('Predicted label', fontsize=16)
plt.ylabel('True label', fontsize=16)
plt.show()