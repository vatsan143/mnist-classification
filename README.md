# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit categorization of scanned handwriting images, together with answer verification.
There are a number of handwritten digits in the MNIST dataset. The assignment is to place a handwritten digit picture into one of ten classes that correspond to integer values from 0 to 9, inclusively. The dataset consists of 60,000 handwritten digits that are each 28 by 28 pixels in size. In this case, we construct a convolutional neural network model that can categorise to the relevant numerical value.
## Neural Network Model

![image](https://github.com/user-attachments/assets/4321be8a-d17b-4405-aa33-6494fec79bd4)



## DESIGN STEPS

### STEP 1: 

Import the required packages

### STEP 2: 

Load the dataset

### STEP 3: 

Scale the dataset

### STEP 4: 

Use the one-hot encoder

### STEP 5: 

Create the model

### STEP 6: 

Compile the model

### STEP 7: 

Fit the model

### STEP 8: 

Make prediction with test data and with an external data
## PROGRAM

### Name:
Developed BY : SRIVATSAN G

Reference no : 212223230216


```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[50]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64, validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('two.png')
type(img)
mg = image.load_img('two.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray') 
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
  ```   
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/a776ddc5-c4c4-45a8-b561-da6b1fc0095f)
![image](https://github.com/user-attachments/assets/61c489bc-9cee-4385-9cce-d39096414708)


### Classification Report

![Screenshot 2024-09-13 114135](https://github.com/user-attachments/assets/54d11c4e-ea36-489a-9cfe-16417568d2de)


### Confusion Matrix


![Screenshot 2024-09-13 114152](https://github.com/user-attachments/assets/0ef82168-c146-4a2c-b985-fcff710549db)


### New Sample Data Prediction
![Screenshot 2024-09-13 114209](https://github.com/user-attachments/assets/7c7352c3-5d21-4948-b2bc-a4a7d9284e2c)




## RESULT
Therefore a model has been successfully created for digit classification using mnist dataset.
