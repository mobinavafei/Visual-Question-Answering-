# CLIP-Based Visual Question Answering (VQA)

This project implements a Visual Question Answering (VQA) system using OpenAI's CLIP model as a multimodal feature extractor and a Multi-Layer Perceptron (MLP) classifier for answer prediction.

## Overview

- CLIP encodes images and questions into a shared embedding space.
- Feature fusion combines visual and textual representations.
- An MLP classifier predicts the final answer.
- Transfer learning is applied using a pretrained CLIP model.
- CLIP remains frozen while only the classification head is trained.

## Model Architecture

Image -> CLIP Image Encoder -> Image Embedding

Question -> CLIP Text Encoder -> Text Embedding

Feature Fusion (Concat + Multiply + Abs Difference)

-> MLP Classifier

-> Predicted Answer

## Dataset Preparation

- Answers were cleaned and standardized.
- The top 300 most frequent answers were selected.
- Rare and noisy answers were removed.
- The task was formulated as a classification problem.

## Training Configuration

- Backbone: CLIP
- Optimizer: Adam
- Learning Rate: 1e-5
- Batch Size: 16
- Loss Function: Cross Entropy Loss
- Training Strategy: Frozen CLIP + Trainable MLP

## Results

- Validation Accuracy: ~29%

## Future Work

- LLM integration for natural language explanations
- Generative VQA architectures
- Attention-based multimodal fusion
- Reinforcement learning for answer refinement

## Technologies Used

Python, PyTorch, CLIP, NumPy, Pandas, Matplotlib, Scikit-learn
