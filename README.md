# Potato Disease Classification using CNN

## Overview

Potato Disease Classification using CNN is a computer vision project aimed at classifying potato plant images into three categories: "Early Blight," "Healthy," and "Late Blight." Leveraging the power of Convolutional Neural Networks (CNNs), this project can help farmers and researchers quickly identify and mitigate disease outbreaks in potato crops. The dataset used for training and validation is sourced from Kaggle, and data augmentation techniques using preprocessing layers from Keras have been applied to improve model generalization.

## Dataset

The project utilizes a comprehensive dataset from Kaggle, consisting of images of potato plants affected by early blight, healthy plants, and late blight. The dataset has been carefully labeled and divided into training and validation sets to enable supervised learning. Here's the link for the dataset https://www.kaggle.com/datasets/arjuntejaswi/plant-village

## Data Augmentation

To enhance the model's ability to generalize, data augmentation techniques have been applied during training using Keras preprocessing layers. Techniques such as rotation, horizontal flipping, and zooming are employed to create variations of the original images, resulting in an augmented dataset that helps prevent overfitting.

## Model Architecture

The classification model is built using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The architecture consists of multiple convolutional layers, followed by pooling and dense layers to make predictions. The model is optimized using appropriate loss functions and metrics for multi-class classification.

## Performance Evaluation

The trained model is evaluated on the validation set to assess its performance in distinguishing between early blight, healthy, and late blight in potato plants. The evaluation metrics provide valuable insights into the model's accuracy, precision, recall, and F1-score for each class.

## How to Use the Project

To utilize the potato disease classification model, follow these steps:

1. Clone this repository to your local machine.

2. Download the Potato Disease Classification Dataset from Kaggle using the provided link.

3. Set up the required dependencies, including TensorFlow, Keras, and other necessary libraries, as detailed in the project's README file.

4. Preprocess the dataset, apply data augmentation techniques, and split it into training and validation sets.

5. Train the CNN model on the augmented dataset using the training script provided.

6. Evaluate the trained model's performance on the validation set and analyze the classification metrics.

7. To use the model for prediction on new images, load the saved model weights and pass the images through the CNN model.
