# Highway OCR & Panel Detector

An Optical Character Recognition (OCR) system designed to automatically locate and read text on highway information panels. This project was developed for the Artificial Vision course at Universidad Rey Juan Carlos (URJC).

The system operates in two main phases: training a supervised classifier to identify alphanumeric characters in the standard **"highway-gothic" traffic font**, and an image processing module that segments characters, groups them into text lines using RANSAC, and recognizes the final text.

---

## 1. Features & Methodology

- **Character Classification:**  
  Evaluated multiple models (LDA + Naive Bayes, PCA + Naive Bayes). The final system uses Linear Discriminant Analysis (LDA) for dimensionality reduction (from 625 to 61 dimensions) combined with a K-Nearest Neighbors (KNN, K=5) classifier. This approach achieved a **95.61% accuracy**.

- **Image Preprocessing:**  
  Uses adaptive thresholding to convert characters to white on a black background, followed by contour detection to isolate bounding boxes.

- **Text Line Grouping:**  
  Uses RANSAC regression to iteratively adjust lines to the centers of detected characters, filtering out extreme diagonals.

- **Full Pipeline Integration:**  
  Integrates with a panel detector (using MSER/Hough and Non-Maximum Suppression) to process full highway images, crop detected signs, and extract text.

---

## 2. Requirements

- **Python 3.8+**

Python libraries:

- `opencv-python`
- `scikit-learn`
- `numpy`
- `matplotlib`

> This is a pure Python project: **no compilation is required.**

### Installation

It is recommended to use a virtual environment:

```bash
# Create and activate the virtual environment
python -m venv .venv

# Activate (Linux / macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install opencv-python scikit-learn numpy matplotlib
```
## 3. Project Structure
.
├── ocr_classifier.py                      # Base OCRClassifier class
├── lda_normal_bayes_classifier.py         # Ex.1: LDA + Naive Bayes
├── lda_normal_knn_classifier.py           # Ex.2: LDA + KNN (Best classifier)
├── pca_normal_bayes_classifier.py         # Ex.2: PCA + Naive Bayes
├── feature_extraction.py                  # Auxiliary feature extraction functions
├── utils_ocr.py                           # Image loading utilities
│
├── evaluar_clasificadores_OCR.py          # Train & evaluate classifiers
├── main_panels_ocr.py                     # Reads text from cropped panels
├── evaluar_resultados_test_ocr_panels.py  # Levenshtein distance evaluation
├── main.py                                # Full system: Detector + OCR
│
├── panel_detection/                       # Panel detector from previous assignment
├── train_ocr/                             # Training data: 0-9, A-Z, a-z (62 classes)
├── test_ocr/                              # Validation/test data
└── IMÁGENES_PANELES/
    ├── test_ocr_panels/                   # Cropped panels + gt.txt
    └── test_detection/                    # Full images + gt.txt

## 4. Usage
All commands should be run from the root of the project.

### Character Classifier Training & Evaluation
Use evaluar_clasificadores_OCR.py to test different classifiers. The script trains on the train_ocr dataset, evaluates on the test dataset, and prints the accuracy along with a confusion matrix.

```bash
# Baseline classifier (LDA + Naive Bayes)
python evaluar_clasificadores_OCR.py --classifier lda_naive_bayes \
       --train_path ./train_ocr --validation_path ./test_ocr

# Best alternative (LDA + KNN)
python evaluar_clasificadores_OCR.py --classifier lda_knn \
       --train_path ./train_ocr --validation_path ./test_ocr

# Second alternative (PCA + Naive Bayes)
python evaluar_clasificadores_OCR.py --classifier pca_naive_bayes \
       --train_path ./train_ocr --validation_path ./test_ocr
```

### Reading Cropped Panels
This step locates characters and lines of text in already cropped panels using the LDA + KNN classifier.

```bash
# 1) Process panels and generate resultado.txt
python main_panels_ocr.py --train_path ./train_ocr \
       --test_path "./IMÁGENES_PANELES/test_ocr_panels"

# 2) Compare with ground truth using Levenshtein distance
python evaluar_resultados_test_ocr_panels.py
```

### Full System Execution (Detection + OCR)
This links the panel detector with the OCR module to process full images.

```bash
python main.py --detector mser --train_path ./train_ocr \
       --test_path "./IMÁGENES_PANELES/test_detection" --visualize_ocr
```

Note: The --visualize_ocr flag opens interactive debugging windows showing character bounding boxes, RANSAC lines, and recognized text. Press any key to advance to the next image.

## 5. Results

The global performance of the system was evaluated using the Levenshtein distance between the recognized text and the ground truth.

- **Best Classifier Accuracy:** 95.61% (LDA + KNN).
- **Panel Reading Performance:** 28 out of 74 panels achieved an edit distance of less than 5.
- **Average Levenshtein Distance:** 7.49 operations.

While the character classifier performs exceptionally well, overall panel reading performance is highly sensitive to panel contrast and noise during the binarization phase.

## 6. Authors

- Isam El Mourabet  
- Santiago Nicolás Díaz Tituaña  
- Cristian Laurentiu Sindila  

**Universidad Rey Juan Carlos (URJC)**  
**Computer Engineering**
