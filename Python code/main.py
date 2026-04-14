# ============================================================
# CRIME HOTSPOT PREDICTION – FULL PIPELINE & MODEL COMPARISON
# Random Forest | XGBoost | LightGBM | CatBoost
# With EDA + ANOVA Feature Selection
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ============================================================
# 1. LOAD DATA
# ============================================================

train_path = r"C:\Users\sowmy\OneDrive\Desktop\Crime_Hotspot_Prediction\data\crime_dataset_2015_to_2020.csv"
test_path  = r"C:\Users\sowmy\OneDrive\Desktop\Crime_Hotspot_Prediction\data\crime_dataset_2020_to_2022.csv"

train_df = pd.read_csv(train_path, low_memory=False)
test_df  = pd.read_csv(test_path, low_memory=False)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# ============================================================
# 2. DATE & TIME FEATURE ENGINEERING
# ============================================================

for df in [train_df, test_df]:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df.dropna(subset=['Date','Latitude','Longitude'], inplace=True)

    df['Hour'] = df['Date'].dt.hour
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS (TRAIN ONLY)
# ============================================================

sns.set_style("whitegrid")

plt.figure(figsize=(10,5))
sns.countplot(data=train_df, x='Hour')
plt.title("Crime Count by Hour")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=train_df, x='DayOfWeek')
plt.title("Crime Count by Day of Week")
plt.show()

plt.figure(figsize=(6,6))
train_df['Arrest'].value_counts().plot(
    kind='pie', autopct='%1.1f%%',
    colors=['lightcoral','lightgreen']
)
plt.title("Arrest Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(12,5))
train_df['Primary Type'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Crime Types")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
sns.kdeplot(
    x=train_df['Longitude'],
    y=train_df['Latitude'],
    fill=True, cmap='Reds', bw_adjust=0.5
)
plt.title("Crime Density Heatmap")
plt.show()

# ============================================================
# 4. HOTSPOT LABEL CREATION (NO DATA LEAKAGE)
# ============================================================

train_df['LatBin'] = (train_df['Latitude'] // 0.01) * 0.01
train_df['LonBin'] = (train_df['Longitude'] // 0.01) * 0.01

grid_counts = (
    train_df.groupby(['LatBin','LonBin'])
    .size()
    .reset_index(name='CrimeCount')
)

train_df = train_df.merge(grid_counts, on=['LatBin','LonBin'], how='left')
threshold = train_df['CrimeCount'].quantile(0.75)
train_df['Hotspot'] = (train_df['CrimeCount'] >= threshold).astype(int)

test_df['LatBin'] = (test_df['Latitude'] // 0.01) * 0.01
test_df['LonBin'] = (test_df['Longitude'] // 0.01) * 0.01
test_df = test_df.merge(grid_counts, on=['LatBin','LonBin'], how='left')
test_df['CrimeCount'].fillna(0, inplace=True)
test_df['Hotspot'] = (test_df['CrimeCount'] >= threshold).astype(int)

# ============================================================
# 5. FEATURE PREPARATION
# ============================================================

features = [
    'Latitude','Longitude',
    'Hour','Month','DayOfWeek','IsWeekend',
    'Community Area','Arrest','Primary Type'
]

train_ml = train_df[features + ['Hotspot']]
test_ml  = test_df[features + ['Hotspot']]

train_ml['Arrest'] = train_ml['Arrest'].astype(int)
test_ml['Arrest']  = test_ml['Arrest'].astype(int)

train_ml = pd.get_dummies(train_ml, columns=['Primary Type'], drop_first=True)
test_ml  = pd.get_dummies(test_ml, columns=['Primary Type'], drop_first=True)

test_ml = test_ml.reindex(columns=train_ml.columns, fill_value=0)

X_train = train_ml.drop('Hotspot', axis=1)
y_train = train_ml['Hotspot']
X_test  = test_ml.drop('Hotspot', axis=1)
y_test  = test_ml['Hotspot']

# ============================================================
# 6. IMPUTATION
# ============================================================

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed  = imputer.transform(X_test)

# ============================================================
# 7. ANOVA FEATURE SELECTION
# ============================================================

k = 20
anova = SelectKBest(score_func=f_classif, k=k)

X_train_selected = anova.fit_transform(X_train_imputed, y_train)
X_test_selected  = anova.transform(X_test_imputed)

anova_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'ANOVA Score': anova.scores_
}).sort_values(by='ANOVA Score', ascending=False)

print("\n========== ANOVA FEATURE RANKING ==========")
print(anova_scores.head(30).to_string(index=False))

selected_features = X_train.columns[anova.get_support()]
print("\nSELECTED FEATURES:")
print(selected_features.tolist())

# ============================================================
# 8. MODELS
# ============================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight='balanced',
        random_state=42, n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_state=42
    )
}

# ============================================================
# 9. TRAIN, EVALUATE & CONFUSION MATRICES
# ============================================================

results = []
roc_data = {}

for name, model in models.items():
    print(f"\n================ {name} =================")

    model.fit(X_train_selected, y_train)

    y_train_pred = model.predict(X_train_selected)
    y_test_pred  = model.predict(X_test_selected)

    y_test_prob  = model.predict_proba(X_test_selected)[:,1]

    results.append([
        name,
        accuracy_score(y_train, y_train_pred),
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred)
    ])

    print("\nTEST CLASSIFICATION REPORT")
    print(classification_report(y_test, y_test_pred))

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    sns.heatmap(confusion_matrix(y_train, y_train_pred),
                annot=True, fmt='d', ax=ax[0])
    ax[0].set_title(f"{name} – Train")

    sns.heatmap(confusion_matrix(y_test, y_test_pred),
                annot=True, fmt='d', ax=ax[1])
    ax[1].set_title(f"{name} – Test")

    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_data[name] = (fpr, tpr, auc(fpr,tpr))

# ============================================================
# 10. ROC CURVE COMPARISON
# ============================================================

plt.figure(figsize=(7,6))
for name,(fpr,tpr,roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ============================================================
# 11. FINAL METRICS TABLE
# ============================================================

results_df = pd.DataFrame(results, columns=[
    'Model','Train Accuracy','Test Accuracy',
    'Precision','Recall','F1 Score'
])

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df)

plt.figure(figsize=(10,5))
sns.barplot(x='Model', y='F1 Score', data=results_df)
plt.title("Model Comparison – Test F1 Score")
plt.ylim(0,1)
plt.show()
