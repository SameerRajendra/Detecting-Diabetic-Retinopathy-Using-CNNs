# 📌 Diabetic Retinopathy Detection using EfficientNet-B4

This project builds a deep learning model to classify Diabetic Retinopathy (DR) stages from retinal fundus images using **EfficientNet-B4**. The pipeline is optimized for **GPU acceleration** using **CuPy** and includes techniques to handle **class imbalance**, improve **generalization**, and accelerate **inference**.

## 📁 Repository Contents

```
├── DR_Classification.ipynb     # Main notebook (training + evaluation)
├── dataset_utils.py            # CuPy-powered dataset loader
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview (this file)
```

## 🚀 Getting Started

1. **Clone this repository:**
```bash
git clone https://github.com/your_username/dr-detection-efficientnet.git
cd dr-detection-efficientnet
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Dataset Structure:**
Dataset linl- https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
```
train/
  ├──1_left.jpeg.png
  ├──5_left.jpeg.png
  └── ...
test/
  └── <test images>
trainLabels.csv
```

4. **Run the Notebook:**
Open `DR_Classification.ipynb` and execute cells sequentially.

---

## 🧠 Model Details

- **Architecture**: EfficientNet-B4
- **Input Size**: 380×380
- **Loss Function**: class weighting
- **Regularization**: Dropout (0.3)
- **Optimizer**: Adam
- **Metrics**: Accuracy, F1-score, Confusion Matrix

---

## 📊 Results

- **Best Validation Accuracy**: ~80%
- **Macro F1-score**: ~0.51
- **Confusion Matrix**: Included in final notebook cell

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- CuPy
- timm
- OpenCV
- scikit-learn
- pandas
- albumentations (optional)

---

## 📄 License

Apache License © 2025
