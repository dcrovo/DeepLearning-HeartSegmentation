# DeepLearning-HeartSegmentation
Heart segmentation in chest CT scans 

## Introduction

This project aims to develop and evaluate deep learning models for the accurate segmentation of the heart in medical images, particularly in non-contrast and non-gated computed tomography (CT) scans. The precise delineation of the heart region is crucial for various medical applications.

## Models and Architectures

### U-Net++

We employed the U-Net++ architecture as a base model for heart segmentation. The model was trained using a Binary Cross-Entropy loss function. After 100 epochs of training, the model achieved an average Dice similarity coefficient of 0.72 and a Jaccard index of 0.67 on the validation set.

### MA-Net

The Multi-scale Attention Net (MA-Net) architecture was implemented to improve heart segmentation. After 100 epochs, this model outperformed U-Net++ with an average Dice coefficient of 0.85 and a Jaccard index of 0.82 on the validation set. Further training for 200 epochs reduced oscillations in the training and validation metrics.

## Results

In summary, the MA-Net model demonstrated superior performance in heart segmentation compared to U-Net++. It achieved a Dice coefficient of 0.883, which compares favorably with state-of-the-art models. However, both models exhibited some overfitting, likely due to the limited size of the training dataset.

## Future Work

To enhance the models further, future work will focus on hyperparameter optimization and increased data augmentation. The well-segmented heart regions will be used as a crucial step for addressing broader medical image analysis challenges.

