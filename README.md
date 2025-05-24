### Problem Statement 
We intend to perform face recognition. Face recognition means that for a given 
image you can tell the subject id. Our database of subject is very simple. It has 40 
subjects. Below we will show the needed steps to achieve the goal of the 
assignment. 

**1. Download the Dataset and Understand the Format**
*a. ORL dataset is available at the following link.* 
https://www.kaggle.com/kasikrit/att-database-of-faces/ 
*b. The dataset has 10 images per 40 subjects. Every image is a grayscale 
image of size 92x112.*

**2. Generate the Data Matrix and the Label vector**
*a. Convert every image into a vector of 10304 values corresponding to 
the image size.*
*b. Stack the 400 vectors into a single Data Matrix D and generate the 
label vector y.*

**The labels are integers from 1:40 corresponding to the subject id.**
**3. Split the Dataset into Training and Test sets**  
*a. From the Data Matrix D400x10304 keep the odd rows for training and 
the even rows for testing. This will give you 5 instances per person for 
training and 5 instances per person for testing.* 
*b. Split the labels vector accordingly.* 
**4. Classification using PCA** 

*a. Use the pseudo code below for computing the projection matrix U. 
Define the alpha = {0.8,0.85,0.9,0.95}* 
*b. Project the training set, and test sets separately using the same 
projection matrix.* 
*c. Use a simple classifier (first Nearest Neighbor to determine the class 
labels).* 
*d. Report Accuracy for every value of alpha separately.*

**5. Classification Using LDA**
*a. Use the pseudo code below for LDA. We will modify few lines in 
pseudocode to handle multiclass LDA.* 
*i. Calculate the mean vector for every class Mu1, Mu2, ..., Mu40.* 
*ii. Replace B matrix by Sb.* 

---

Here, m is the number of classes, 𝜇 is the overall sample mean, and 𝑛𝑘 is the 
number of samples in the k-th class. 

iii. S matrix remains the same, but it sums S1, S2, S3, ...S40. 

iv. Use 39 dominant eigenvectors instead of just one. You will 
have a projection matrix U39x10304. 

b. Project the training set, and test sets separately using the same 
projection matrix U. You will have 39 dimensions in the new space. 
c. Use a simple classifier (first Nearest Neighbor to determine the class 
labels). 
d. Report Accuracy for the Multiclass LDA on the face recognition 
dataset. 
e. Compare the results to PCA results. 

**6. Classifier Tuning** 
a. Set the number of neighbors in the K-NN classifier to 1,3,5,7. 
b. Tie breaking at your preferred strategy. 
c. Plot (or tabulate) the performance measure (accuracy) against the 
K value. This is to be done for PCA and LDA as well. 

---

# Face Recognition System with ORL Dataset

## 📌 Overview
This project implements a complete face recognition system using the ORL dataset, comparing Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) approaches. The system includes performance evaluation and a bonus face vs. non-face classifier.

## ✨ Features
- **Two Recognition Methods**: PCA and LDA for face recognition
- **Performance Comparison**: Detailed accuracy metrics for both methods
- **Interactive Interface**: Streamlit-based web app with visual results
- **Bonus Classifier**: Face vs. non-face discrimination
- **Comprehensive Analysis**: Includes confusion matrices, variance plots, and sample classifications

## 🛠️ Technical Implementation
```python
import streamlit as st
import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.linalg import eigh
import zipfile
import tempfile
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler

# [Rest of the imports and code...]
```

## 🚀 How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run face_recognition.py
```

3. Upload the ORL dataset ZIP file when prompted

## 📊 Results Preview
![PCA vs LDA Comparison](https://example.com/path/to/screenshot.png) *(example image)*

## 📂 Project Structure
```
face-recognition/
├── face_recognition.py      # Main application code
├── requirements.txt         # Dependencies
├── README.md                # This file
└── assets/                  # Sample images/screenshots
```

## 🔧 Dependencies
- Python 3.8+
- Streamlit
- scikit-learn
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- Pillow

## 📝 License
MIT License

---


