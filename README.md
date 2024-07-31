# A-Deep-Learning-Approach-for-Multi-Label-Chest-X-ray-Diagnosis-Using-DenseNet-121


This repository contains the code and resources for training a DenseNet-121 model to classify chest X-rays into 14 distinct thoracic pathologies. The model is based on the DenseNet-121 architecture and is trained on the NIH Chest X-ray dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Model Saving](#model-saving)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project uses a DenseNet-121 model pre-trained on ImageNet to classify chest X-ray images into 14 different thoracic diseases. The training process includes data augmentation, using TensorFlow's TPU strategy for faster computation, and various callbacks to optimize the model.

## Installation

To run the code in this repository, you will need to have Python 3.x installed along with the following libraries:

- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
You can install these libraries using pip:

pip install tensorflow keras pandas numpy matplotlib


## Dataset

The dataset used for training the model is the NIH Chest X-ray dataset. It contains 10,000 chest X-ray images, each labeled with one or more of 14 distinct thoracic pathologies. The dataset can be downloaded from the following link: [NIH Chest X-ray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).

## Model Training

To train the model,  run the following file : model-training-densenet-121.ipynb

This will train the DenseNet-121 model on the NIH Chest X-ray dataset and save the trained model. The model is trained for 100 epochs with a batch size of 32 and a learning rate of 0.001. The model is trained using the TPU strategy for faster computation. The model is also trained with data augmentation and various callbacks to optimize the model.

## Results

The trained model can be used to classify new chest X-ray images into the 14 distinct thoracic pathologies.
This will output the top 5 predicted pathologies along with their corresponding probabilities.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improving the code, please open an issue or submit a pull request.
