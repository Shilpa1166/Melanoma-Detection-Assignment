Melanoma Detection using Custom CNN Model
This project involves developing a custom convolutional neural network (CNN) model in TensorFlow to accurately detect melanoma from images of skin lesions. The goal is to aid dermatologists by automating the detection process, thereby reducing manual effort and improving early diagnosis.

General Information
Background: Melanoma is a deadly form of skin cancer accounting for 75% of skin cancer deaths. Early detection is crucial for effective treatment.

Business Problem: The project aims to develop a solution that can evaluate images and alert dermatologists about the presence of melanoma, reducing manual effort in diagnosis.

Dataset: The dataset consists of 2357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). It includes images of nine types of skin diseases: Actinic keratosis, Basal cell carcinoma, Dermatofibroma, Melanoma, Nevus, Pigmented benign keratosis, Seborrheic keratosis, Squamous cell carcinoma, and Vascular lesion.

Technologies Used
TensorFlow - version 2.x (for building and training the CNN model)
Augmentor - version 0.2.10 (for handling class imbalances)
Google Colab - (for GPU acceleration during model training)

Conclusions
Model Performance: The custom CNN model showed promising results in detecting melanoma, with improvements observed after implementing data augmentation and addressing class imbalances.

Data Augmentation: Applying data augmentation techniques effectively resolved issues of overfitting and underfitting, enhancing model robustness.

Class Imbalance Rectification: Using the Augmentor library to rectify class imbalances significantly improved model performance on minority classes.

