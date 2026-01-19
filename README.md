## Obesity Level Classification using Artificial Neural Networks

## This project focuses on predicting obesity levels based on individuals’ physical characteristics and lifestyle habits using an Artificial Neural Network (ANN).

The study was developed as part of an **Introduction to Artificial Neural Networks** course and demonstrates the full machine learning pipeline, including data preprocessing, model design, training, and performance evaluation.

---

## 📌 Project Overview

Obesity is a major global health issue influenced by multiple interrelated factors such as age, weight, eating habits, physical activity, and daily lifestyle choices.
Due to the complex and non-linear nature of these relationships, traditional methods may struggle to provide accurate predictions.

In this project, a **multi-class classification model** based on **Artificial Neural Networks** was developed to predict obesity levels with high accuracy.

---

## 🚀 Features

* Multi-class obesity level classification
* Comprehensive data preprocessing pipeline
* Deep neural network with dropout regularization
* Hyperparameter tuning and early stopping
* High classification performance on test data

---

## 🧠 Model Architecture

The model was built using the **Keras Sequential API** and consists of:

* **Input Layer:** Fully connected layer with 256 neurons (ReLU)
* **Hidden Layers:**

  * 128 neurons (ReLU + Dropout 30%)
  * 64 neurons (ReLU + Dropout 20%)
* **Output Layer:** 7 neurons (Softmax) representing obesity classes

**Optimizer:** Adam
**Loss Function:** Categorical Crossentropy
**Evaluation Metric:** Accuracy

---

## 📊 Performance

* **Test Accuracy:** **95.27%**
* Strong class separation observed in the confusion matrix
* High AUC values in ROC analysis, indicating robust classification capability

---

## 🖼️ Project Result & Grade

Below is an image showing the evaluation/grade received for this project:

> ℹ️ Replace `images/grade.png` with the actual path of your image in the repository.

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries & Tools:**

  * TensorFlow
  * Keras
  * NumPy
  * Pandas
  * Scikit-learn
  * Matplotlib

---

## 📂 Dataset

* **Source:** Kaggle – Obesity Levels Dataset
* The dataset includes both numerical and categorical features such as:

  * Age, Height, Weight
  * Eating habits
  * Physical activity level
  * Daily lifestyle behaviors

All categorical variables were encoded, and numerical features were standardized before training.

---

## 📌 Notes

* This repository focuses on **model development and evaluation**.
* Raw datasets and large output files are intentionally excluded.
* The project is intended for **educational and portfolio purposes**.

---

## 👤 Author

**Anıl Han Aydın**
Computer Engineering Student

---

## 📜 License

This project is provided for academic and educational use.
