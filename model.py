# MCC Improved Classifier with SHAP, Realistic Metrics, Stage Prediction

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
# STEP 1: Load Dataset
print("\U0001F504 Loading dataset...")
df = pd.read_csv("dataset.csv", index_col=0)

# Diagnostic: Print index and shape to verify format
print("\n🔍 BEFORE transpose")
print("Shape:", df.shape)
print("Index sample:", df.index[:5])
print("Columns sample:", df.columns[:5])

# Transpose dataset to get samples as rows if genes are rows
if df.index[0].startswith("VP") or df.index[0].startswith("RP"):
    df = df.T
    print("\n📐 Transposed dataset")
    print("Shape:", df.shape)
    print("Index (samples):", df.index[:5])

# STEP 2: Create MCC label from barcode suffix (-1 = PBMC, -2 = Tumor)
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

df['MCC'] = df.index.to_series().apply(lambda x: 1 if str(x).endswith('-2') else 0)

# Optional: Group barcodes to avoid leakage (e.g., by barcode suffix)
df['SampleGroup'] = df.index.to_series().astype(str).str.extract(r'(.*)-\d$')[0]
df = df[df['SampleGroup'].notna()]
print("✅ Group extraction successful. SampleGroup count:", df['SampleGroup'].nunique())

# STEP 3: Prepare features and labels
X = df.drop(columns=['MCC', 'SampleGroup'])
X = X.select_dtypes(include=['number'])
y = df['MCC']
groups = df['SampleGroup']

# STEP 4: Grouped Train-Test Split to prevent data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# STEP 5: Train Random Forest Classifier
print("\n\U0001F680 Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# STEP 6: Evaluate Model Performance
y_pred = clf.predict(X_test)
print("\n✅ MCC Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('MCC Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# STEP 7: SHAP Explanation for MCC Model
print("\n\U0001F4CA SHAP Summary Plot...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

try:
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, max_display=20)
    else:
        shap.summary_plot(shap_values, X_test, max_display=20)
except Exception as e:
    print("⚠ SHAP summary plot failed:", e)

# STEP 8: Simulated Stage Classification (if real stage not provided)
print("\n\U0001F3AF Simulating Stage Prediction...")
df_stage = df[df['MCC'] == 1].copy()
df_stage['Stage'] = np.random.choice(['Stage I', 'Stage III', 'Stage IV'], size=len(df_stage))

stage_X = df_stage.drop(columns=['MCC', 'SampleGroup', 'Stage'])
stage_X = stage_X.select_dtypes(include=['number'])
stage_y = df_stage['Stage']

le = LabelEncoder()
y_stage = le.fit_transform(stage_y)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(stage_X, y_stage, test_size=0.2, stratify=stage_y, random_state=42)

stage_clf = RandomForestClassifier(n_estimators=100, random_state=42)
stage_clf.fit(X_train_s, y_train_s)
y_pred_s = stage_clf.predict(X_test_s)

print("\n✅ Stage Classification Report:")
print(classification_report(y_test_s, y_pred_s, target_names=le.classes_))

cm_s = confusion_matrix(y_test_s, y_pred_s)
sns.heatmap(cm_s, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Stage Prediction Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# SHAP for Stage Classification
explainer_stage = shap.TreeExplainer(stage_clf)
shap_values_stage = explainer_stage.shap_values(X_test_s)

try:
    if isinstance(shap_values_stage, list):
        shap.summary_plot(shap_values_stage[0], X_test_s, max_display=20)
    else:
        shap.summary_plot(shap_values_stage, X_test_s, max_display=20)
except Exception as e:
    print("⚠ SHAP stage plot failed:", e)

print("\n✅ Done. All tasks completed.")