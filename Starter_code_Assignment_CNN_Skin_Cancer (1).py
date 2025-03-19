#!/usr/bin/env python
# coding: utf-8

# Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

# ### Importing Skin Cancer Data
# #### To do: Take necessary actions to read the data

# ### Importing all the important libraries

# In[1]:


import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[11]:


pip install numpy --upgrade --force-reinstall


# In[2]:


pip install tensorflow


# In[13]:


pip install daal==2021.2.3


# In[ ]:


## If you are using the data by mounting the google drive, use the following :
## from google.colab import drive
## drive.mount('/content/gdrive')

##Ref:https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166


# This assignment uses a dataset of about 2357 images of skin cancer types. The dataset contains 9 sub-directories in each train and test subdirectories. The 9 sub-directories contains the images of 9 skin cancer types respectively.

# In[5]:


# Defining the path for train and test images
## Todo: Update the paths of the train and test dataset
pip install pathlib
data_dir_train = pathlib.Path("https://drive.google.com/drive/folders/1AizlYUmHmg6migFJzv1aU7d1vsjTVF_p")
data_dir_test = pathlib.Path("https://drive.google.com/drive/folders/1G6GzAvPycwrFIwzzrrG2TCx8Qmwx9aMN")


# In[ ]:


image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
print(image_count_train)
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print(image_count_test)


# ### Load using keras.preprocessing
# 
# Let's load these images off disk using the helpful image_dataset_from_directory utility.

# ### Create a dataset
# 
# Define some parameters for the loader:

# In[ ]:


batch_size = 32
img_height = 180
img_width = 180


# Use 80% of the images for training, and 20% for validation.

# In[ ]:


## Write your train dataset here
## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# In[ ]:


## Write your validation dataset here
## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# In[ ]:


# List out all the classes of skin cancer and store them in a list. 
# You can find the class names in the class_names attribute on these datasets. 
# These correspond to the directory names in alphabetical order.
class_names = train_ds.class_names
print(class_names)


# ### Visualize the data
# #### Todo, create a code to visualize one instance of all the nine classes present in the dataset

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.

# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
# 
# `Dataset.prefetch()` overlaps data preprocessing and model execution while training.

# In[ ]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ### Create the model
# #### Todo: Create a CNN model, which can accurately detect 9 classes present in the dataset. Use ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network. Here, it is good to standardize values to be in the `[0, 1]`

# In[ ]:


model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.summary()


# ### Compile the model
# Choose an appropirate optimiser and loss function for model training 

# In[ ]:


### Todo, choose an appropirate optimiser and loss function

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[ ]:


# View the summary of all layers
model.summary()


# ### Train the model

# In[ ]:


epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# ### Visualizing training results

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### Todo: Write your findings after the model fit, see if there is an evidence of model overfit or underfit

# ### Write your findings here

# In[ ]:


# Todo, after you have analysed the model fit history for presence of underfit or overfit, choose an appropriate data augumentation strategy. 
# Your code goes here
train_datagen_augmented = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator_augmented = train_datagen_augmented.flow_from_directory(
    data_dir_train,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_generator_augmented = train_datagen_augmented.flow_from_directory(
    data_dir_train,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# In[1]:


# Todo, visualize how your augmentation strategy works for one instance of training image.
# Your code goes here
plt.figure(figsize=(10, 10))
for images, _ in train_generator_augmented.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()


# ### Todo:
# ### Create the model, compile and train the model
# 

# In[ ]:


## You can use Dropout layer if there is an evidence of overfitting in your findings

## Your code goes here
model_augmented = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Adding Dropout to prevent overfitting
    layers.Dense(len(class_names), activation='softmax')
])




# ### Compiling the model

# In[ ]:


## Your code goes here
model_augmented.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])


# ### Training the model

# In[ ]:


## Your code goes here, note: train your model for 20 epochs


history_augmented = model_augmented.fit(
    train_generator_augmented,
    validation_data=validation_generator_augmented,
    epochs=20
)


# ### Visualizing the results

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### Todo: Write your findings after the model fit, see if there is an evidence of model overfit or underfit. Do you think there is some improvement now as compared to the previous model run?

# #### **Todo:** Find the distribution of classes in the training dataset.
# #### **Context:** Many times real life datasets can have class imbalance, one class can have proportionately higher number of samples compared to the others. Class imbalance can have a detrimental effect on the final model quality. Hence as a sanity check it becomes important to check what is the distribution of classes in the data.

# In[ ]:


## Your code goes here.
class_counts = train_df['label'].value_counts()
print(class_counts)

least_samples_class = class_counts.idxmin()
dominant_classes = class_counts.idxmax()
print(f"Class with least samples: {least_samples_class}")
print(f"Dominant classes: {dominant_classes}")


# #### **Todo:** Write your findings here: 
# #### - Which class has the least number of samples?
# #### - Which classes dominate the data in terms proportionate number of samples?
# 

# #### **Todo:** Rectify the class imbalance
# #### **Context:** You can use a python package known as `Augmentor` (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

# In[ ]:


get_ipython().system('pip install Augmentor')

import Augmentor

path_to_training_dataset = "path_to_training_dataset"
for class_name in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + class_name)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)

image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(image_count_train)


# To use `Augmentor`, the following general procedure is followed:
# 
# 1. Instantiate a `Pipeline` object pointing to a directory containing your initial image data set.<br>
# 2. Define a number of operations to perform on this data set using your `Pipeline` object.<br>
# 3. Execute these operations by calling the `Pipeline’s` `sample()` method.
# 

# In[ ]:


path_to_training_dataset="To do"
import Augmentor
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500) ## We are adding 500 samples per class to make sure that none of the classes are sparse.


# Augmentor has stored the augmented images in the output sub-directory of each of the sub-directories of skin cancer types.. Lets take a look at total count of augmented images.

# In[ ]:


image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(image_count_train)


# ### Lets see the distribution of augmented data after adding new images to the original training data.

# In[ ]:


path_list = [x for x in glob(os.path.join(data_dir_train, '*','output', '*.jpg'))]
path_list


# In[ ]:


lesion_list_new = [os.path.basename(os.path.dirname(os.path.dirname(y))) for y in glob(os.path.join(data_dir_train, '*','output', '*.jpg'))]
lesion_list_new


# In[ ]:


dataframe_dict_new = dict(zip(path_list_new, lesion_list_new))


# In[ ]:


df2 = pd.DataFrame(list(dataframe_dict_new.items()),columns = ['Path','Label'])
new_df = original_df.append(df2)


# In[ ]:


new_df['Label'].value_counts()


# So, now we have added 500 images to all the classes to maintain some class balance. We can add more images as we want to improve training process.

# #### **Todo**: Train the model on the data created using Augmentor

# In[ ]:


batch_size = 32
img_height = 180
img_width = 180


# #### **Todo:** Create a training dataset

# In[ ]:


data_dir_train="path to directory with training data + data created using augmentor"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = ## Todo choose the correct parameter value, so that only training data is refered to,,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# #### **Todo:** Create a validation dataset

# In[ ]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=123,
  validation_split = 0.2,
  subset = ## Todo choose the correct parameter value, so that only validation data is refered to,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# #### **Todo:** Create your model (make sure to include normalization)

# In[2]:


## your code goes here
model_balanced = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Adding Dropout to prevent overfitting
    layers.Dense(len(class_names), activation='softmax')
])


# #### **Todo:** Compile your model (Choose optimizer and loss function appropriately)

# In[ ]:


## your code goes here
model_balanced.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['accuracy'])


# #### **Todo:**  Train your model

# In[ ]:


epochs = 50
history_balanced = model_balanced.fit(
    train_ds_balanced,
    validation_data=val_ds_balanced,
    epochs=epochs
)


# #### **Todo:**  Visualize the model results

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# #### **Todo:**  Analyze your results here. Did you get rid of underfitting/overfitting? Did class rebalance help?
# 
# 

# After visualizing the results, analyze the following:
# 
# Did you get rid of underfitting/overfitting?: Compare the training and validation accuracy and loss curves. If the curves are close to each other and both show improvement, it indicates that the model is neither underfitting nor overfitting.
# Did class rebalance help?: Check if the validation accuracy has improved after handling class imbalances. If the validation accuracy is higher and the model generalizes better, it indicates that class rebalancing was effective.
