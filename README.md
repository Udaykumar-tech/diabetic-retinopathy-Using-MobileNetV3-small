# 🩺 Diabetic Retinopathy Detection Using MobileNetV3-Small + CBAM + Ensemble Learning 

A clinically aligned deep-learning system for automated 5-class diabetic retinopathy (DR) classification, optimized for real-time deployment using TensorFlow Lite.  
The system integrates MobileNetV3-Small, CBAM attention, and ensemble learning to deliver high accuracy, fast inference, and strong interpretability suitable for medical decision support.

---

## 📑 Abstract

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness globally.  
This project presents a lightweight, clinically reliable, and interpretable DR detection system capable of classifying fundus images into five DR severity levels.

Using MobileNetV3-Small, CBAM attention, and an ensemble classifier, the system achieves:

- **89.8% Training Accuracy**  
- **69.1% Validation Accuracy**  
- **92% DR Detection Rate**  
- **13 MB TensorFlow Lite model**  
- **51 ms inference time**  
- **High interpretability via Grad-CAM**  

Optimized for edge deployment, telemedicine, and real-time clinical screening.

---

## ⚕️ 1. Clinical Background

| Class | Severity      | Description                          |
|------|----------------|---------------------------------------|
| 0    | No DR          | No abnormalities                      |
| 1    | Mild NPDR      | Early microaneurysms                  |
| 2    | Moderate NPDR  | Increasing vascular changes           |
| 3    | Severe NPDR    | Extensive hemorrhages                 |
| 4    | PDR            | Neovascularization; sight-threatening |

---

## 🗂️ 2. Datasets

### **APTOS 2019**
- 3,662 labeled images  
- 5-class clinician grading  

### **EyePACS**
- 35,126 training images  
- 53,579 test images  

### **Preprocessing Pipeline**
- Multi-dataset integration  
- Image quality enhancement  
- **Input Size: 224×224**  
- Flip, rotation, brightness/contrast augmentations  
- Class-balanced sampling  

---

## 🧠 3. Methodology

### ✔️ Transfer Learning  
MobileNetV3-Small with ImageNet weights.

### ✔️ CBAM Attention Module  
Enhances feature focus via:
- Channel attention  
- Spatial attention  

### ✔️ Ensemble Classifier  
Weighted averaging of multiple heads → robust predictions.

### ✔️ Optimizations  
- ReduceLROnPlateau  
- EarlyStopping  
- L2 Regularization  
- Gradient Accumulation  

---

## 🏗️ 4. System Architecture
Input (224×224 RGB)
       │
 MobileNetV3-Small
       │
     CBAM
       │
 Ensemble Classifier
       │
Softmax Output (5 Classes)

### Deployment Features
- **Model Size:** 13 MB  
- **Format:** TFLite (.tflite)  
- **Latency:** ~51 ms  
- Supports laptops, mobile devices & edge systems  

---

## 🏋️‍♂️ 5. Training Summary

| Parameter        | Value |
|------------------|-------|
| Epochs           | 100 (best @ 82) |
| Batch Size       | 32 |
| Optimizer        | Adam |
| Input Resolution | 224×224 |
| Loss Function    | Sparse Categorical Crossentropy |
| LR Scheduler     | Adaptive |

---

## 📊 6. Results & Performance

| Metric               | Value |
|----------------------|-------|
| Training Accuracy    | **89.8%** |
| Validation Accuracy  | **69.1%** |
| DR Detection Rate    | **92%** |
| Inference Time       | **51 ms** |
| Model Size           | **13 MB** |

### Explainability
- Grad-CAM heatmaps  
- CBAM attention maps  
- Fine-grained confidence scores  

---

## 💻 7. Usage Instructions

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Train the Model
```bash
python train.py
```
### Make Predictions
```bash
python predict.py --image path/to/image.jpg
````
### Convert to TensorFlow Lite
```bash
python convert_model.py
```
### Evaluate
```bash
python evaluate.py
```

## 📁 8. Directory Structure
### 📦 diabetic-retinopathy-detection
#### ┣ 📂 data/
#### ┣ 📂 models/
#### ┣ 📂 reports/
#### ┣ 📂 src/
#### ┣ 📜 train.py
#### ┣ 📜 predict.py
#### ┣ 📜 convert_model.py
#### ┣ 📜 evaluate.py
#### ┣ 📜 requirements.txt
#### ┗ 📜 README.md

## 🔮 9. Future Enhancements
### Near-Term

Improved validation performance

Enhanced class balancing

Confidence calibration

### Long-Term

Multi-modal medical data integration

Federated learning

Real-time integration with fundus cameras

Disease progression tracking

## 🏁 Conclusion

This project delivers a clinically interpretable, computationally efficient, and deployment-ready diabetic retinopathy detection system.
With MobileNetV3-Small, CBAM, and TFLite optimization, the model balances accuracy, speed, and explainability—critical for real-world medical AI adoption. Do it for it 
