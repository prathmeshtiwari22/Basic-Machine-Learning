# Basic Machine Learning Projects

Welcome to the collection of basic machine learning projects! This repository includes several classic machine learning tasks with detailed implementations. Each project explores different aspects of machine learning and showcases practical applications. Below is a brief description of each project included in this repository.

## Projects

### 1. Breast Cancer Classification

**Description**: This project aims to predict whether a tumor is malignant or benign based on various features extracted from breast cancer samples. The dataset used is the Breast Cancer Wisconsin dataset, which is widely used for classification tasks.

- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Algorithms Used**: Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest
- **Features**: Mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Usage**: Classify tumors and evaluate model performance using various metrics.

**How to Run**:
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python breast_cancer_classification.py`

### 2. Credit Card Fraud Detection

**Description**: This project focuses on detecting fraudulent transactions using a credit card transactions dataset. It employs anomaly detection techniques to identify suspicious activities.

- **Dataset**: Credit Card Fraud Detection Dataset
- **Algorithms Used**: Isolation Forest, Local Outlier Factor, One-Class SVM
- **Features**: Transaction amount, time of transaction, anonymized features
- **Metrics**: Precision, Recall, F1 Score, ROC Curve
- **Usage**: Detect and evaluate fraudulent transactions.

**How to Run**:
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python credit_card_fraud_detection.py`

### 3. Face Recognition

**Description**: This project aims to recognize and identify individuals from facial images. It uses various face recognition techniques and pre-trained models.

- **Dataset**: Custom dataset or pre-existing facial datasets like LFW (Labeled Faces in the Wild)
- **Algorithms Used**: Convolutional Neural Networks (CNNs), FaceNet, OpenCV
- **Features**: Facial landmarks, embeddings
- **Metrics**: Accuracy, Precision, Recall
- **Usage**: Recognize and verify faces from images or video streams.

**How to Run**:
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python face_recognition.py`

### 4. Phishing Website Detection

**Description**: This project detects phishing websites using features extracted from website URLs and HTML content. It helps in identifying potentially harmful websites.

- **Dataset**: Phishing Websites Dataset
- **Algorithms Used**: Random Forest, Decision Trees, Naive Bayes
- **Features**: URL length, number of special characters, presence of HTTPS
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Usage**: Classify websites as phishing or legitimate.

**How to Run**:
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python phishing_website_detection.py`

### 5. Board Review Prediction

**Description**: This project predicts the outcome of board reviews based on historical data and various features related to the review process.

- **Dataset**: Board Review Dataset
- **Algorithms Used**: Linear Regression, Logistic Regression, Gradient Boosting
- **Features**: Review score, reviewer experience, number of papers reviewed
- **Metrics**: Mean Absolute Error, Mean Squared Error, R-squared
- **Usage**: Predict board review outcomes and assess model performance.

**How to Run**:
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python board_review_prediction.py`

### 6. Titanic Survival Prediction

**Description**: This classic project aims to predict survival on the Titanic based on passenger information. It's a well-known dataset for binary classification tasks.

- **Dataset**: Titanic Dataset
- **Algorithms Used**: Logistic Regression, Random Forest, K-Nearest Neighbors (KNN)
- **Features**: Age, Sex, Fare, Cabin, Embarked
- **Metrics**: Accuracy, Precision, Recall, F1 Score
- **Usage**: Predict survival status of passengers and evaluate model performance.

**How to Run**:
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python titanic_survival_prediction.py`

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/prathmeshtiwari22/Basic-Machine-Learning.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Basic-Machine-Learning
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Acknowledgments

- The datasets used in these projects are publicly available and provided by various organizations.
- Special thanks to the open-source community for their contributions to machine learning libraries and tools.

