# SVM Spam Detection Classifier

This project demonstrates a **Support Vector Machine (SVM) Classifier** for detecting spam emails. It uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features and then trains an SVM model to classify emails as either "Spam" or "Not Spam."

## Overview

Spam detection is essential for email service providers to filter out unwanted content. This project aims to implement a machine learning-based approach using SVM, a supervised learning algorithm ideal for binary classification tasks.

The classifier takes in text data from email samples, converts it into TF-IDF vectors, and uses an SVM to classify the emails. The model is trained with a small, sample dataset provided within the code.

## Features

- **Text Preprocessing**: Converts email text data into TF-IDF features for model compatibility.
- **Binary Classification**: SVM model classifies emails as either "Spam" or "Not Spam."
- **Stratified Train-Test Split**: Ensures balanced class representation in training and testing sets.
- **Evaluation Metrics**: Provides classification report and accuracy score to evaluate model performance.

## Requirements

- Python 3.x
- scikit-learn
- pandas

Install the required libraries with:

```bash
pip install scikit-learn pandas
