# Ultrasound Image Data Augmentation using GAN for Non-invasive White Blood Cell Counting

This project aims to enhance the non-invasive white blood cell counting device, **Neosonics**, by creating synthetic in vitro ultrasound images to address the issue of data scarcity in medical imaging. We developed a **Generative Adversarial Network (GAN)** to augment ultrasound images of peritoneal dialysis, aiming to improve the training of deep learning models for more efficient image generation.

## Overview

The goal of this project is to use GAN-based data augmentation techniques to generate synthetic ultrasound images. These images will help in increasing both the **volume** and **variability** of training samples, which are crucial for enhancing deep learning models' performance in non-invasive white blood cell counting.

Key techniques employed:
- **Generative Adversarial Networks (GANs)**: Conditional GAN models to generate diverse, high-quality ultrasound images of varying white blood cell concentrations.
- **Wasserstein GAN**: Implemented to overcome mode collapse and ensure stable image generation.
- **Data Preprocessing**: Involves various transformations to enhance the visualization of particle traces in ultrasound images.

## Features

- **Data Augmentation**: Uses GAN to generate synthetic ultrasound images for different white blood cell concentrations.
- **GAN Models Implemented**:
  - Conditional GAN (C-GAN)
  - Auxiliary Classifier GAN (AC-GAN)
  - Least Squares GAN (LSGAN)
  - Wasserstein GAN (WGAN)
- **Image Preprocessing**: Includes Butterworth filtering, dynamic range adjustment, and Sobel filtering to highlight particle traces.



