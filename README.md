# Breast Cancer Histopathology Classifier

## Overview

This project uses deep learning and transfer learning to classify breast histopathology image patches as either:

* Non-Cancer
* Cancer

The project was developed as an end-to-end machine learning application, including:

* Data preprocessing
* Exploratory data analysis
* CNN baseline model
* MobileNetV2 transfer learning model
* Model evaluation
* Streamlit web application

The final system allows users to upload histopathology image patches and receive real-time cancer predictions.

---

## Dataset

Dataset: Breast Histopathology Images (IDC Regular Patches)

Statistics:

* Total Images: 277,524
* Non-Cancer Images: 198,738
* Cancer Images: 78,786

Image Size:

* 50 × 50 pixels
* RGB color images

Class Distribution:

| Class      |   Count | Percentage |
| ---------- | ------: | ---------: |
| Non-Cancer | 198,738 |      71.6% |
| Cancer     |  78,786 |      28.4% |

The dataset is imbalanced, making cancer recall an important evaluation metric.

## Data Processing Pipeline

1. Load histopathology image patches
2. Extract labels from filenames
3. Train/Validation/Test split
4. TensorFlow Dataset creation
5. Image normalization
6. Batch generation
7. Model training

Dataset Split:

| Dataset    |  Images |
| ---------- | ------: |
| Train      | 194,266 |
| Validation |  41,629 |
| Test       |  41,629 |

---

## Model 1: Baseline CNN

Architecture:

* Conv2D (32 filters)
* MaxPooling
* Conv2D (64 filters)
* MaxPooling
* Conv2D (128 filters)
* MaxPooling
* Dense Layer
* Dropout
* Sigmoid Output

### Results

| Metric          |  Value |
| --------------- | -----: |
| Test Accuracy   | 82.45% |
| Cancer Recall   |    45% |
| Cancer F1 Score |   0.58 |

Confusion Matrix:

* True Positives: 247
* False Negatives: 296

The baseline model achieved good overall accuracy but missed a large number of cancer samples.

---

## Model 2: MobileNetV2 Transfer Learning

Transfer learning was used to improve cancer detection performance.

Configuration:

* MobileNetV2 pretrained on ImageNet
* Global Average Pooling
* Dense Layer
* Dropout
* Sigmoid Output

### Results

| Metric          | Value |
| --------------- | ----: |
| Test Accuracy   | 80.0% |
| Cancer Recall   |   82% |
| Cancer F1 Score |  0.69 |

Confusion Matrix:

* True Positives: 443
* False Negatives: 100

Although overall accuracy decreased slightly, cancer recall improved significantly.

---

## Model Comparison

| Metric          |   CNN | MobileNetV2 |
| --------------- | ----: | ----------: |
| Accuracy        | 82.4% |       80.0% |
| Cancer Recall   |   45% |         82% |
| Cancer F1 Score |  0.58 |        0.69 |

The MobileNetV2 model was selected as the final model because it reduced missed cancer detections by approximately 66%.

---

## Streamlit Application

Features:

* Upload histopathology image
* Real-time prediction
* Cancer probability score
* Interactive user interface

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Streamlit
* Git
* GitHub
