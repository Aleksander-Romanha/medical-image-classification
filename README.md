# Medical Image Classification: Logistic Regression & kNN

This repository contains an applied machine learning research project exploring classification algorithms for breast ultrasound images.

## 📌 Project Overview

The objective of this project is to classify breast ultrasound images into two distinct categories: **Cancer** (negative class) and **Normal** (positive class). 

We evaluate the performance of classical Logistic Regression alongside a non-parametric K-Nearest Neighbors (kNN) classifier to establish strong baselines for medical image analysis.

### Key Highlights:
- **Logistic Regression:** Investigating an unregularized baseline model.
- **Regularization:** Applying **L2 Regularization (Ridge)** and performing hyperparameter grid search to mitigate overfitting in a high-dimensional feature space (784 features per image).
- **Class Imbalance:** Implementing class-weight balancing strategies to penalize minority-class misclassifications, evaluated using Balanced Accuracy and ROC-AUC metrics.
- **K-Nearest Neighbors (kNN):** Optimizing the neighborhood size ($k$) with distance-based weighting. Given the high variance and dimensionality, $k=1$ proved to be the optimal parameter.

## 🛠️ Technologies Used
- **Python 3**
- **Scikit-Learn** (Logistic Regression, kNN, Metrics)
- **PyTorch & Torchvision** (Data loading and augmentation)
- **MedMNIST API**
- **Matplotlib & Pandas**

## 📊 Dataset
We use the **BreastMNIST** dataset from the MedMNIST collection, consisting of 780 grayscale images, each with a resolution of 28x28 pixels. The images are pre-divided into training, validation, and test sets.

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/medical-image-classification.git
   cd medical-image-classification
   ```

2. Activate your virtual environment and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook efc2_classification_knn_logistic.ipynb
   ```

## 📈 Results
- The application of **L2 Regularization** significantly improved generalization over the unregularized logistic regression baseline.
- **Class Balancing** further improved the Balanced Accuracy and F1-Score, ensuring the model does not become biased toward the majority class.
- The **1-Nearest Neighbor (1-NN)** classifier provided a robust baseline, outperforming some linear models due to the non-linear, high-variance nature of the raw pixel space.
