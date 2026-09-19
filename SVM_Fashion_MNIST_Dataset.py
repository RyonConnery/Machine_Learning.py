# ============================================================
# SVM Classifier
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD THE FASHION-MNIST DATASET
# ============================================================

# Load the complete Fashion-MNIST dataset required by the assessment.
# Each observation represents one 28 x 28 grayscale clothing image
# flattened into 784 pixel values.
print("Loading Fashion-MNIST dataset...")

X, y = fetch_openml(
    "Fashion-MNIST",
    version=1,
    return_X_y=True,
    as_frame=False
)

print("\nFASHION-MNIST LOADED SUCCESSFULLY")
print("-" * 60)

print(f"Total images: {X.shape[0]:,}")
print(f"Pixels per image: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")


# ============================================================
# 2. CREATE THE REQUIRED TRAINING AND TESTING SPLIT
# ============================================================

# Split the complete dataset using the 80/20 configuration
# specified in the Capella assessment instructions.
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nFULL DATASET SPLIT")
print("-" * 60)

print(f"Full training set: {len(X_train_full):,} images")
print(f"Full testing set: {len(X_test_full):,} images")


# ============================================================
# 3. CREATE REPRESENTATIVE WORKING SUBSETS
# ============================================================

# Kernel SVM training becomes computationally expensive on tens of
# thousands of 784-feature images. Representative stratified subsets
# are therefore used for the required kernel and hyperparameter
# experiments while preserving all 10 Fashion-MNIST classes.

X_train, _, y_train, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=10000,
    random_state=42,
    stratify=y_train_full
)

X_test, _, y_test, _ = train_test_split(
    X_test_full,
    y_test_full,
    train_size=2000,
    random_state=42,
    stratify=y_test_full
)

print("\nSVM WORKING SUBSETS")
print("-" * 60)

print(f"SVM training images: {len(X_train):,}")
print(f"SVM testing images: {len(X_test):,}")


# ============================================================
# 4. NORMALIZE PIXEL VALUES
# ============================================================

# Fashion-MNIST pixels range from 0 to 255.
# Scale them to 0-1 because SVM performance depends strongly
# on the scale of the numerical features.
X_train = X_train.astype("float64") / 255.0
X_test = X_test.astype("float64") / 255.0

print("\nPIXEL NORMALIZATION")
print("-" * 60)

print(
    f"Training pixel range: "
    f"{X_train.min():.1f} to {X_train.max():.1f}"
)

print(
    f"Testing pixel range: "
    f"{X_test.min():.1f} to {X_test.max():.1f}"
)


# ============================================================
# 5. LINEAR-KERNEL SVM
# ============================================================

# A linear kernel creates linear decision boundaries in the
# high-dimensional image feature space.
print("\nTraining Linear SVM...")

svm_linear = SVC(
    kernel="linear",
    C=1.0
)

svm_linear.fit(X_train, y_train)

linear_pred = svm_linear.predict(X_test)

linear_accuracy = accuracy_score(
    y_test,
    linear_pred
)

print(
    f"Linear Kernel Accuracy (C=1.0): "
    f"{linear_accuracy:.4f}"
)


# ============================================================
# 6. RBF-KERNEL SVM
# ============================================================

# The RBF kernel allows nonlinear decision boundaries.
# gamma='scale' automatically adjusts gamma to the data scale.
print("\nTraining RBF SVM...")

svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_rbf.fit(X_train, y_train)

rbf_pred = svm_rbf.predict(X_test)

rbf_accuracy = accuracy_score(
    y_test,
    rbf_pred
)

print(
    f"RBF Kernel Accuracy "
    f"(C=1.0, gamma='scale'): "
    f"{rbf_accuracy:.4f}"
)


# ============================================================
# 7. POLYNOMIAL-KERNEL SVM
# ============================================================

# The polynomial kernel models nonlinear relationships through
# polynomial combinations of the image pixel features.
print("\nTraining Polynomial SVM...")

svm_poly = SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    gamma="scale"
)

svm_poly.fit(X_train, y_train)

poly_pred = svm_poly.predict(X_test)

poly_accuracy = accuracy_score(
    y_test,
    poly_pred
)

print(
    f"Polynomial Kernel Accuracy "
    f"(degree=3, C=1.0): "
    f"{poly_accuracy:.4f}"
)


# ============================================================
# 8. RBF HYPERPARAMETER EXPERIMENT
# ============================================================

# Increase C from 1.0 to 10.0 to examine how a lower degree
# of regularization affects classification accuracy.
print("\nTraining RBF SVM with C=10.0...")

svm_rbf_c10 = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale"
)

svm_rbf_c10.fit(X_train, y_train)

rbf_c10_pred = svm_rbf_c10.predict(X_test)

rbf_c10_accuracy = accuracy_score(
    y_test,
    rbf_c10_pred
)

print(
    f"RBF Kernel Accuracy "
    f"(C=10.0, gamma='scale'): "
    f"{rbf_c10_accuracy:.4f}"
)


# ============================================================
# 9. COMPARE ALL SVM CONFIGURATIONS
# ============================================================

results = pd.DataFrame({
    "Configuration": [
        "Linear: C=1.0",
        "RBF: C=1.0, gamma=scale",
        "Polynomial: degree=3, C=1.0",
        "RBF: C=10.0, gamma=scale"
    ],
    "Accuracy": [
        linear_accuracy,
        rbf_accuracy,
        poly_accuracy,
        rbf_c10_accuracy
    ]
})

results = results.sort_values(
    "Accuracy",
    ascending=False
).reset_index(drop=True)

print("\nSVM PERFORMANCE COMPARISON")
print("-" * 60)

display(
    results.style.format({
        "Accuracy": "{:.4f}"
    })
)


# ============================================================
# 10. FINAL INTERPRETATION
# ============================================================

best_model = results.loc[0, "Configuration"]
best_accuracy = results.loc[0, "Accuracy"]

print("\nKEY FINDINGS")
print("-" * 60)

print(
    f"1. The highest accuracy was {best_accuracy:.4f}, "
    f"achieved by {best_model}."
)

print(
    f"2. Linear kernel accuracy: "
    f"{linear_accuracy:.4f}."
)

print(
    f"3. RBF kernel accuracy with C=1.0: "
    f"{rbf_accuracy:.4f}."
)

print(
    f"4. Polynomial kernel accuracy: "
    f"{poly_accuracy:.4f}."
)

print(
    f"5. RBF kernel accuracy with C=10.0: "
    f"{rbf_c10_accuracy:.4f}."
)

print(
    "6. Kernel choice affected accuracy because linear and "
    "nonlinear kernels create different decision boundaries."
)

print(
    "7. Changing C altered the balance between regularization "
    "and fitting the training observations."
)

print(
    "8. The representative subset preserved all ten Fashion-MNIST "
    "classes while making the required SVM experiments practical "
    "within the notebook environment."
)

print(
    "\nAssessment 5 SVM classification completed successfully."
)
