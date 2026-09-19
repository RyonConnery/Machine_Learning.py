# ============================================================
# Comparing Model Accuracy With and Without PCA
# ============================================================

# Import the libraries required for data loading, feature scaling,
# dimensionality reduction, classification, evaluation, and visualization.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD THE BREAST CANCER WISCONSIN DATASET
# ============================================================

# Load the diagnostic Breast Cancer Wisconsin dataset.
# The dataset contains 30 numerical predictor features and
# two target classes: malignant and benign.
data = load_breast_cancer()

X = data.data
y = data.target

print("BREAST CANCER DATASET LOADED SUCCESSFULLY")
print("-" * 65)

print(f"Number of observations: {X.shape[0]}")
print(f"Original number of features: {X.shape[1]}")
print(f"Target classes: {list(data.target_names)}")


# ============================================================
# 2. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Use the exact 80/20 split and random state specified
# in Capella's supplemental assessment instructions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING AND TESTING SETS")
print("-" * 65)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 3. STANDARDIZE THE FEATURES
# ============================================================

# PCA is sensitive to differences in feature scale.
# StandardScaler transforms each training feature to have
# approximately mean 0 and standard deviation 1.
#
# The scaler is fitted only on the training data to avoid
# allowing information from the test set to influence training.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nSTANDARDIZATION")
print("-" * 65)

print(
    f"Mean of standardized training features: "
    f"{X_train_scaled.mean():.4f}"
)

print(
    f"Standard deviation of standardized training features: "
    f"{X_train_scaled.std():.4f}"
)


# ============================================================
# 4. APPLY PCA
# ============================================================

# Reduce the original 30 standardized predictors to exactly
# two principal components, as required by the assessment.
pca = PCA(
    n_components=2
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

print("\nPRINCIPAL COMPONENT ANALYSIS")
print("-" * 65)

print(
    f"Original feature count: "
    f"{X_train_scaled.shape[1]}"
)

print(
    f"PCA feature count: "
    f"{X_train_pca.shape[1]}"
)

# explained_variance_ratio_ shows how much of the original
# standardized-data variance each principal component retains.
explained_variance = pca.explained_variance_ratio_

total_explained_variance = (
    explained_variance.sum()
)

print(
    f"Principal Component 1 explained variance: "
    f"{explained_variance[0]:.4f}"
)

print(
    f"Principal Component 2 explained variance: "
    f"{explained_variance[1]:.4f}"
)

print(
    f"Total variance retained by two components: "
    f"{total_explained_variance:.4f}"
)


# ============================================================
# 5. VISUALIZE THE TWO PRINCIPAL COMPONENTS
# ============================================================

# The two-component representation allows the 30-dimensional
# dataset to be visualized on a two-dimensional scatter plot.
plt.figure(figsize=(9, 6))

for target_value, target_name in enumerate(data.target_names):
    mask = y_train == target_value

    plt.scatter(
        X_train_pca[mask, 0],
        X_train_pca[mask, 1],
        label=target_name,
        alpha=0.70
    )

plt.title(
    "Breast Cancer Wisconsin Data After PCA"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 6. TRAIN LOGISTIC REGRESSION WITHOUT PCA
# ============================================================

# Train Logistic Regression using all 30 standardized features.
# This model serves as the baseline for comparison.
model_no_pca = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model_no_pca.fit(
    X_train_scaled,
    y_train
)

predictions_no_pca = model_no_pca.predict(
    X_test_scaled
)

accuracy_no_pca = accuracy_score(
    y_test,
    predictions_no_pca
)

print("\nMODEL WITHOUT PCA")
print("-" * 65)

print(
    f"Accuracy without PCA: "
    f"{accuracy_no_pca:.4f}"
)


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION WITH PCA
# ============================================================

# Train the same type of classifier using only the two
# principal components produced by PCA.
model_pca = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model_pca.fit(
    X_train_pca,
    y_train
)

predictions_pca = model_pca.predict(
    X_test_pca
)

accuracy_pca = accuracy_score(
    y_test,
    predictions_pca
)

print("\nMODEL WITH PCA")
print("-" * 65)

print(
    f"Accuracy with PCA: "
    f"{accuracy_pca:.4f}"
)


# ============================================================
# 8. COMPARE MODEL ACCURACY
# ============================================================

accuracy_difference = (
    accuracy_pca - accuracy_no_pca
)

comparison_df = pd.DataFrame({
    "Model": [
        "Logistic Regression Without PCA",
        "Logistic Regression With PCA"
    ],
    "Number of Features": [
        X_train_scaled.shape[1],
        X_train_pca.shape[1]
    ],
    "Accuracy": [
        accuracy_no_pca,
        accuracy_pca
    ]
})

print("\nMODEL ACCURACY COMPARISON")
print("-" * 65)

display(
    comparison_df.style.format({
        "Accuracy": "{:.4f}"
    })
)

print(
    f"Accuracy Improvement with PCA: "
    f"{accuracy_difference:.4f}"
)


# ============================================================
# 9. VISUALIZE THE ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    [
        "Without PCA\n30 Features",
        "With PCA\n2 Components"
    ],
    [
        accuracy_no_pca,
        accuracy_pca
    ]
)

plt.title(
    "Logistic Regression Accuracy With and Without PCA"
)

plt.ylabel("Accuracy")
plt.ylim(0, 1.0)

plt.tight_layout()
plt.show()


# ============================================================
# 10. SUMMARIZE THE RESULTS
# ============================================================

print("\nKEY FINDINGS")
print("-" * 65)

print(
    f"1. PCA reduced the feature space from "
    f"{X_train_scaled.shape[1]} features to "
    f"{X_train_pca.shape[1]} principal components."
)

print(
    f"2. The two principal components retained "
    f"{total_explained_variance:.4f} of the standardized "
    f"dataset's total variance."
)

print(
    f"3. Logistic Regression accuracy without PCA was "
    f"{accuracy_no_pca:.4f}."
)

print(
    f"4. Logistic Regression accuracy with PCA was "
    f"{accuracy_pca:.4f}."
)

print(
    f"5. The accuracy difference after PCA was "
    f"{accuracy_difference:.4f}."
)

if accuracy_difference > 0:
    print(
        "6. In this experiment, PCA improved classification "
        "accuracy while reducing dimensionality."
    )
elif accuracy_difference < 0:
    print(
        "6. In this experiment, PCA slightly reduced classification "
        "accuracy, demonstrating the trade-off between dimensionality "
        "reduction and retained predictive information."
    )
else:
    print(
        "6. In this experiment, PCA preserved the same classification "
        "accuracy while substantially reducing dimensionality."
    )

print(
    "7. PCA can reduce redundancy among correlated predictors "
    "by representing the original variables through orthogonal "
    "principal components."
)

print(
    "\nAssessment 8 PCA accuracy comparison completed successfully."
)
