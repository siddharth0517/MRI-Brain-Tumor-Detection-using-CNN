# MRI Brain Tumor Detection Using CNN
This project involves the classification of human brain MRI images into four categories—glioma, meningioma, no tumor, and pituitary tumor—using Convolutional Neural Networks (CNNs). The dataset consists of MRI images sourced from Kaggle and aims to help identify brain tumors accurately using deep learning techniques.

## Dataset
The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It consists of 7023 brain MRI images grouped into the following categories:

+ Glioma
+ Meningioma
+ No Tumor
+ Pituitary Tumor
The images are provided in .jpeg format and are labeled accordingly. These MRI images are ideal for building a classification model to identify the presence and type of brain tumors.

## Project Workflow
### 1. Data Preprocessing
+ Image resizing: All MRI images are resized to a fixed size to maintain uniformity for the CNN model.
+ Data augmentation: Techniques such as rotation, flipping, and scaling are used to augment the dataset and improve model generalization.
+ Train-test split: The dataset is divided into training and validation sets to evaluate the performance of the model.
### 2. CNN Architecture
+ The model used for this project is a Convolutional Neural Network (CNN) that includes multiple convolutional layers followed by pooling layers and fully connected layers.
+ Optimizer: Adam optimizer is used for faster convergence.
+ Loss function: Categorical cross-entropy is employed as the loss function for this multi-class classification task.
### 3. Model Training
+ The CNN model is trained over multiple epochs, with a batch size of 32.
### 4. Evaluation Metrics
+ Accuracy: The overall accuracy of the model on the validation set.
+ Confusion Matrix: Provides insights into the model’s classification performance for each tumor type.
+ Precision, Recall, and F1-score: These metrics are also calculated to assess the model's performance in classifying brain tumors.

### 5. Visualization
Sample MRI images along with their predicted labels are visualized using matplotlib to showcase the model's predictions.
![download](https://github.com/user-attachments/assets/3ba9df26-d915-4d31-af4e-763921445de2)


### Results
The CNN model achieved an overall accuracy of 65% on the validation set. Additional evaluation metrics, including precision, recall, and F1-score, further confirm the model's ability to classify brain tumors accurately.

### Requirements
+ Python 3.6 or above
+ TensorFlow 2.x
+ Keras
+ Matplotlib
+ NumPy
+ scikit-learn

### Author
Siddharth Jaiswal
