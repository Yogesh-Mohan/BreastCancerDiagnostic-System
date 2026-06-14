import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for saving charts without GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Ensure directories exist
os.makedirs('dataset', exist_ok=True)
os.makedirs('static/images/charts', exist_ok=True)

# 1. Load and process dataset
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()

# We map feature names from 'mean radius' style to 'radius_mean' style
feature_mapping = {
    'mean radius': 'radius_mean',
    'mean texture': 'texture_mean',
    'mean perimeter': 'perimeter_mean',
    'mean area': 'area_mean',
    'mean smoothness': 'smoothness_mean',
    'mean compactness': 'compactness_mean',
    'mean concavity': 'concavity_mean',
    'mean concave points': 'concave_points_mean',
    'mean symmetry': 'symmetry_mean',
    'mean fractal dimension': 'fractal_dimension_mean'
}

# Extract only the 10 mean features
selected_features = list(feature_mapping.keys())
X_raw = pd.DataFrame(data.data, columns=data.feature_names)[selected_features]
X_raw.rename(columns=feature_mapping, inplace=True)

# Target mapping: sklearn has 0=malignant, 1=benign.
# We reverse it so that 1=Malignant (positive) and 0=Benign (negative) for standard clinical convention.
y_raw = 1 - data.target

# Combine and save as CSV
df = X_raw.copy()
df['target'] = y_raw
df.to_csv('dataset/breast_cancer.csv', index=False)
print("Dataset successfully saved to dataset/breast_cancer.csv")

# Display dataset details
print("\nDataset Info:")
print(df.info())
print("\nDataset Descriptive Statistics:")
print(df.describe())

# 2. Split dataset
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Fitted StandardScaler saved to scaler.pkl")

# 4. Generate Visualizations
print("\nGenerating visual plots...")
# A. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/images/charts/correlation_heatmap.png')
plt.close()

# B. Class Distribution Chart
plt.figure(figsize=(6, 5))
sns.countplot(x='target', data=df, palette=['#198754', '#dc3545'])
plt.title('Target Class Distribution')
plt.xticks(ticks=[0, 1], labels=['Benign (0)', 'Malignant (1)'])
plt.xlabel('Diagnosis Class')
plt.ylabel('Patient Count')
plt.tight_layout()
plt.savefig('static/images/charts/class_distribution.png')
plt.close()

# 5. Model Training & Evaluation
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42) # probability=True for confidence percentage
}

best_model_name = None
best_model_obj = None
best_accuracy = 0.0
model_results = []

print("\nModel Training and Evaluation:")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    model_results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    })
    
    # Save best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model_obj = model

# C. Save Model Performance Chart for Admin/About
results_df = pd.DataFrame(model_results)
plt.figure(figsize=(8, 5))
results_melted = pd.melt(results_df, id_vars='Model', var_name='Metric', value_name='Value')
sns.barplot(x='Model', y='Value', hue='Metric', data=results_melted, palette='viridis')
plt.title('Machine Learning Model Comparison')
plt.ylim(0.7, 1.05)
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('static/images/charts/model_comparison.png')
plt.close()

# D. Feature Importance Chart (for best tree model / RF or coefficients of SVM/Logistic)
plt.figure(figsize=(10, 6))
if best_model_name in ['Random Forest', 'Decision Tree']:
    importances = best_model_obj.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_sorted = [X.columns[i] for i in indices]
    sns.barplot(x=importances[indices], y=features_sorted, palette='mako')
    plt.title(f'Feature Importance ({best_model_name})')
elif best_model_name in ['Logistic Regression', 'Support Vector Machine']:
    # Use Logistic Regression coefficients or absolute linear coefficients for feature rank
    if best_model_name == 'Logistic Regression':
        coefs = np.abs(best_model_obj.coef_[0])
    else:
        # For non-linear SVM, we can fall back to using Logistic Regression to show a general linear importance proxy
        temp_lr = LogisticRegression().fit(X_train_scaled, y_train)
        coefs = np.abs(temp_lr.coef_[0])
    indices = np.argsort(coefs)[::-1]
    features_sorted = [X.columns[i] for i in indices]
    sns.barplot(x=coefs[indices], y=features_sorted, palette='mako')
    plt.title(f'Feature Importance Proxy (Scaled Coefficients)')
else:
    plt.title('Feature Importance')
plt.xlabel('Importance Value')
plt.tight_layout()
plt.savefig('static/images/charts/feature_importance.png')
plt.close()

print(f"\n--> Best Performing Model: {best_model_name} with test accuracy {best_accuracy:.4f}")

# 6. Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)
print(f"Best model ({best_model_name}) successfully saved to model.pkl")
