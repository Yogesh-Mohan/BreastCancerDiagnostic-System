# Breast Cancer Diagnostic System (BCDS)

OncoShield BCDS is a clinical decision-support web application that predicts breast mass tumor malignancy using machine learning classifiers. The system runs on Flask and performs real-time classification as **Benign** or **Malignant** based on cytological cell nuclei measurements.

---

## Key Features
- **Machine Learning Prognosis**: Uses an optimized Support Vector Machine (SVM) model trained on the Breast Cancer Wisconsin dataset, achieving a **95.6% diagnostic validation accuracy**.
- **Interactive Triage Input**: User-friendly web form accepting the 10 main cytological cell nuclei features with real-time range checks to minimize typing errors.
- **SQL Database Integration**: Automatically logs diagnosis transactions, patient metadata, and feature parameters into an SQLite database (`predictions.db`).
- **Administrative Control Panel**: Clean dashboard providing search filters (by patient name or ID), diagnostic filters, and CSV spreadsheet exports.
- **On-Demand PDF Reports**: Download professional, structured clinical PDF diagnostic reports for any patient with one click.
- **Clean Responsive UI**: Modern Bootstrap 5 healthcare dashboard optimized for both desktop and mobile medical tablets.

---

## Project Folder Structure
```
BreastCancerDiagnostic/
│
├── app.py                     # Main Flask web application server
├── train_model.py             # Script to load data, train ML models, and save pickle
├── model.pkl                  # Serialized best trained model (SVM)
├── scaler.pkl                 # Serialized fitted StandardScaler
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive system documentation
│
├── dataset/
│   └── breast_cancer.csv      # Locally saved Wisconsin Breast Cancer CSV dataset
│
├── static/
│   ├── css/
│   │   └── style.css          # Custom medical dashboard styles and animations
│   ├── js/
│   │   └── main.js            # Frontend validation and warning bounds handler
│   └── images/
│       └── charts/            # Saved evaluation plots (Heatmap, accuracy stats)
│
└── templates/
    ├── base.html              # Shared layout skeleton
    ├── index.html             # System Overview dashboard
    ├── predict.html           # Diagnostic input form
    ├── result.html            # Diagnostic analysis results
    ├── admin.html             # Administrative records database
    └── about.html             # Clinical info about the dataset & models
```

---

## Dataset & Features
The model is trained on the **Wisconsin Breast Cancer FNA Dataset**. To make the input process quick and simple for medical staff, the system uses the 10 core **mean cell nuclei** features:

1. **Radius Mean**: Mean of distances from center to perimeter points.
2. **Texture Mean**: Standard deviation of gray-scale values.
3. **Perimeter Mean**: Cell nuclei boundary circumference length.
4. **Area Mean**: Surface area of the cell nucleus.
5. **Smoothness Mean**: Local variation in cell radius lengths.
6. **Compactness Mean**: Measures circular complexity ($\text{perimeter}^2 / \text{area} - 1.0$).
7. **Concavity Mean**: Severity of nucleus boundary indentations.
8. **Concave Points Mean**: Number of concave portions along the contour.
9. **Symmetry Mean**: Symmetry comparison of nuclear hemispheres.
10. **Fractal Dimension Mean**: Nuclear boundary complexity (coastline approximation).

---

## Machine Learning Model Comparison
During training (`train_model.py`), four classification algorithms were trained and validated on a 20% test set split:

| Model | Validation Accuracy | F1-Score | Recall (Sensitivity) |
| :--- | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **95.61%** | **93.83%** | **90.48%** |
| **Random Forest** | 93.86% | 91.76% | 92.86% |
| **Logistic Regression** | 92.98% | 90.70% | 92.86% |
| **Decision Tree** | 89.47% | 85.37% | 83.33% |

*Note: For cancer triage diagnostics, maximizing **Recall** (reducing False Negatives) is critical to prevent missing malignant cases. SVM has the highest overall accuracy and validation F1-Score.*

---

## Installation & Setup

### Prerequisites
- Python 3.8 to 3.11 installed.

### Steps
1. **Clone or Navigate to the Directory**:
   ```bash
   cd BreastCancerDiagnostic
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Models**:
   Run the training script to generate the serialized files (`model.pkl` and `scaler.pkl`) and diagnostic charts.
   ```bash
   python train_model.py
   ```

4. **Launch the Web App**:
   Start the Flask development server:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Database Architecture
Patient records are persisted in `predictions.db` (SQLite) under the table `patient_predictions`:
- `id`: Unique identifier (Primary Key).
- `timestamp`: Date and time of diagnostic run.
- `patient_name`: Input patient name.
- `patient_id`: Clinical reference ID.
- `radius_mean` to `fractal_dimension_mean`: Numerical cell inputs.
- `prediction`: Benign or Malignant categorization output.
- `confidence`: Prognostic probability percentage.

---

## Clinical Triage Disclaimer
OncoShield BCDS is an educational decision-support triage tool and does not constitute formal medical or pathological advice. All malignant classifications must be verified through clinical tissue biopsy procedures before treatment planning.

---

## Future Roadmap
- **Deep Learning Classifier**: Integration of a multi-layer perceptron (MLP) neural network model.
- **RESTful API**: Expose `/api/predict` endpoints for integration with Electronic Health Record (EHR) systems.
- **Image FNA Upload**: Upload digitized FNA images directly for automated cell detection and metric parsing.
