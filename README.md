# Image Colorization with Autoencoders

## Overview

This repository contains code for image colorization using convolutional autoencoders. Autoencoders are neural network architectures that can be trained to learn efficient representations of input data. In the context of image colorization, we leverage autoencoders to predict color information from black and white images.

## Motivation

Autoencoders can be tricked during training by introducing slight variations to the input images. By doing this, the autoencoder learns to generate color variations of the training set, resulting in artificial colorization of black and white images.

## Project Structure

- `data_setup.py`: Custom dataset loader for handling and preparing image data.
- `engine.py`: File containing various training functions.
- `model.py`: Definition of the autoencoder model.
- `train.py`: Main script for training and colorizing images.
- `utils.py`: Utility functions for saving trained model.

## Requirements

Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
