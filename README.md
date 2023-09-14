# Hateful Memes Identification with Multimodal Classification

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Models](#models)
  - [1. Baseline Model](#baseline-model)
  - [2. VisualBert Model](#visualbert-model)
  - [3. CLIP Fine Tuned Model](#clip-fine-tuned-model)

## Introduction

Welcome to the GitHub repository for the "Hateful Memes Identification with Multimodal Classification" project. In this project, we have addressed the pressing issue of identifying hateful memes in social media content using multimodal classification techniques. Our approach combines textual and visual information from memes and leverages cutting-edge models to achieve this task.

## Problem Statement

The proliferation of hateful content on social media platforms is a growing concern. Memes, often used to spread hate and misinformation, pose a significant challenge for content moderation. The primary goal of this project is to automatically identify hateful memes within social media posts. We aim to classify memes into two categories: hateful and non-hateful.

## Models

We have engineered and evaluated three different multimodal architectures by leveraging state-of-the-art models to address the hateful meme identification problem:

### 1. Baseline Model

- **Model Description**: This model utilizes ResNet, a powerful convolutional neural network, to extract features from meme images.
- **Textual Component**: Text associated with memes is processed using Sentence BERT.
- **Fusion Technique**: Early fusion is employed to combine image and text embeddings.

### 2. VisualBert Model

- **Model Description**: This model combines BERT Tokenizer for text and Region-based Convolutional Neural Network (R-CNN) for image processing.
- **Textual Component**: BERT Tokenizer encodes textual content.
- **Visual Component**: R-CNN extracts features from meme images.
- **Fusion Technique**: Early fusion technique was used to combine BERT Tokenizer and R-CNN embeddings and fed to VisualBert Model for classification.

### 3. CLIP Fine Tuned Model

- **Model Description**: This architecture used state-of-the-art model CLIP, for both textual and visual feature extraction.
- **Training Strategy**: CLIP was fine-tuned for the hateful meme identification task.

## Dataset

We used Facebook’s Hateful Meme Dataset for training and evaluation.

## Results

Utilizing the early fusion technique to combine text and visual embeddings, we attained a peak AUC score of 0.67 for our models.

  
