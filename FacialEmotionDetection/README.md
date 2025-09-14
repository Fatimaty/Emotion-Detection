# Facial Emotion Detection (KDEF, 4 Emotions)

This project implements a **custom CNN** to classify facial expressions into 4 emotions using the **KDEF dataset**.  
It was developed from scratch without using pretrained models.

---

## 📂 Repository Structure
- `model.py` → Custom CNN model definition.  
- `train.py` → Training pipeline for KDEF dataset.  
- `predict.py` → Run inference on a single image.  
- `utils.py` → Preprocessing and dataset loading functions.  
- `requirements.txt` → Required Python packages.  

---

## 🚀 Usage
### 1. Train
```bash
python train.py --data /path/to/KDEF --epochs 20
```

### 2. Predict
```bash
python predict.py --image sample.jpg --weights best_model.pth
```

---

## 🧠 Techniques Used
- Custom CNN architecture (no pretrained models).  
- Dataset: KDEF (4 emotions).  
- Framework: PyTorch.  

---

## 📊 Results
Achieved good performance on 4 basic emotions.  
(Add accuracy/F1 score if you want here.)