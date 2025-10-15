# 🩺 Diabetic Retinopathy Detection Using MobileNetV3-Small

## 📘 Overview
Diabetic Retinopathy (DR) is a serious complication of diabetes that damages the retina and can lead to permanent vision loss if undiagnosed.  
This project applies **Deep Learning** using the **MobileNetV3-Small** architecture to detect and classify DR severity levels from retinal fundus images.  

To ensure robust and generalizable performance:
- The **APTOS 2019 Blindness Detection** dataset is used for **model training**.  
- The **EyePACS** dataset is used for **cross-validation**.

---

## 🎯 Objective
- To build a lightweight, efficient, and accurate model for DR detection.  
- To classify images into 5 stages of Diabetic Retinopathy.  
- To validate the model across different datasets (APTOS → EyePACS).  
- To enable mobile and cloud-based screening applications for remote diagnosis.

---

## ⚙️ Model Architecture – MobileNetV3-Small
**MobileNetV3-Small** is a compact CNN model by Google optimized for **speed and accuracy on low-power devices**.  
It uses:
- **Depthwise Separable Convolutions**
- **Squeeze-and-Excitation (SE) Blocks**
- **Hard-Swish (h-swish) Activation**
- **Inverted Residuals**
- **Neural Architecture Search (NAS)** optimization  

This makes it ideal for **medical image analysis on mobile or edge devices**.

---

## 🧠 Methodology

### 1. Data Collection
- **Training Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
- **Cross-Validation Dataset:** [EyePACS Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- Both datasets contain **retinal fundus images** categorized into 5 DR severity levels:
  1. No DR  
  2. Mild  
  3. Moderate  
  4. Severe  
  5. Proliferative DR  

### 2. Data Preprocessing
- Resize images to **224×224** pixels  
- Normalize pixel values (0–1 scaling)  
- Apply **Data Augmentation**:
  - Rotation  
  - Horizontal & Vertical Flip  
  - Zoom & Brightness Adjustment  
- One-hot encode labels for multi-class classification

### 3. Model Development
- Base Model: `MobileNetV3-Small` pre-trained on **ImageNet**
- Custom classification head:
  ```python
  GlobalAveragePooling2D → Dense(128, activation='relu') → Dropout(0.3) → Dense(5, activation='softmax')
