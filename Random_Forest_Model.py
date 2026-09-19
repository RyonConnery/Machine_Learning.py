# ============================================================
# Random Forest Model
# ============================================================

# Import the libraries required to load the dataset, organize
# the data, train the two classifiers, evaluate predictions,
# calculate feature importance, and create visualizations.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD AND EXPLORE THE BREAST CANCER DATASET
# ============================================================

# Load Scikit-Learn's Breast Cancer Wisconsin diagnostic dataset.
# The target represents the two diagnostic classes:
# 0 = malignant and 1 = benign.
data = load_breast_cancer()

X = data.data
y = data.target

print("BREAST CANCER DATASET LOADED SUCCESSFULLY")
print("-" * 65)

print(f"Number of observations: {X.shape[0]}")
print(f"Number of predictor features: {X.shape[1]}")
print(f"Target classes: {list(data.target_names)}")

# Convert the predictor matrix into a DataFrame so the feature
# names and first observations can be inspected clearly.
df = pd.DataFrame(
    X,
    columns=data.feature_names
)

df["target"] = y

print("\nFirst five rows:")
display(df.head())


# ============================================================
# 2. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Use the exact 70/30 split and random state specified in
# Capella's supplemental assessment instructions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nTRAINING AND TESTING SETS")
print("-" * 65)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 3. TRAIN THE RANDOM FOREST CLASSIFIER
# ============================================================

# The Random Forest contains 100 individual decision trees.
# Each tree contributes to the ensemble prediction, which reduces
# dependence on any one tree's particular structure.
forest_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_clf.fit(
    X_train,
    y_train
)

# Evaluate the fitted Random Forest on the unseen testing set.
forest_predictions = forest_clf.predict(X_test)

forest_accuracy = accuracy_score(
    y_test,
    forest_predictions
)

print("\nRANDOM FOREST PERFORMANCE")
print("-" * 65)

print(
    f"Random Forest Accuracy: "
    f"{forest_accuracy:.4f}"
)


# ============================================================
# 4. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

# feature_importances_ measures each predictor's contribution
# to impurity reduction across all trees in the forest.
forest_importances = forest_clf.feature_importances_

# Sort the features from highest to lowest Random Forest importance.
forest_indices = np.argsort(
    forest_importances
)[::-1]

forest_importance_df = pd.DataFrame({
    "Feature": data.feature_names[forest_indices],
    "Random Forest Importance": forest_importances[forest_indices]
})

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print("-" * 65)

display(
    forest_importance_df.style.format({
        "Random Forest Importance": "{:.4f}"
    })
)

# Visualize the Random Forest importance distribution.
plt.figure(figsize=(14, 7))

plt.bar(
    range(X_train.shape[1]),
    forest_importances[forest_indices],
    align="center"
)

plt.title(
    "Random Forest Feature Importances"
)

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(
    range(X_train.shape[1]),
    data.feature_names[forest_indices],
    rotation=90
)

plt.tight_layout()
plt.show()


# ============================================================
# 5. TRAIN A SINGLE DECISION TREE
# ============================================================

# Train the single Decision Tree required for comparison with
# the ensemble Random Forest.
tree_clf = DecisionTreeClassifier(
    random_state=42
)

tree_clf.fit(
    X_train,
    y_train
)

# Evaluate the single tree on the same testing observations.
tree_predictions = tree_clf.predict(X_test)

tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

print("\nDECISION TREE PERFORMANCE")
print("-" * 65)

print(
    f"Decision Tree Accuracy: "
    f"{tree_accuracy:.4f}"
)


# ============================================================
# 6. SINGLE-TREE FEATURE IMPORTANCE
# ============================================================

# Extract importance scores from the single fitted tree.
tree_importances = tree_clf.feature_importances_

# Display the Decision Tree values in the SAME feature order used
# for the Random Forest so the two models can be compared directly.
tree_importance_df = pd.DataFrame({
    "Feature": data.feature_names[forest_indices],
    "Decision Tree Importance": tree_importances[forest_indices]
})

print("\nDECISION TREE FEATURE IMPORTANCE")
print("-" * 65)

display(
    tree_importance_df.style.format({
        "Decision Tree Importance": "{:.4f}"
    })
)

plt.figure(figsize=(14, 7))

plt.bar(
    range(X_train.shape[1]),
    tree_importances[forest_indices],
    align="center"
)

plt.title(
    "Decision Tree Feature Importances"
)

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(
    range(X_train.shape[1]),
    data.feature_names[forest_indices],
    rotation=90
)

plt.tight_layout()
plt.show()


# ============================================================
# 7. DIRECTLY COMPARE FEATURE IMPORTANCES
# ============================================================

# Put both models' importance values into one table using the same
# feature order. This makes it possible to see whether the Random
# Forest distributes importance across more predictors than the
# single Decision Tree.
comparison_df = pd.DataFrame({
    "Feature": data.feature_names[forest_indices],
    "Random Forest Importance": forest_importances[forest_indices],
    "Decision Tree Importance": tree_importances[forest_indices]
})

print("\nFEATURE IMPORTANCE COMPARISON")
print("-" * 65)

display(
    comparison_df.style.format({
        "Random Forest Importance": "{:.4f}",
        "Decision Tree Importance": "{:.4f}"
    })
)


# ============================================================
# 8. VISUALIZE THE COMPARISON
# ============================================================

# A side-by-side chart shows how importance is distributed across
# features by the Random Forest versus the individual Decision Tree.
x_positions = np.arange(
    len(data.feature_names)
)

bar_width = 0.40

plt.figure(figsize=(15, 8))

plt.bar(
    x_positions - bar_width / 2,
    forest_importances[forest_indices],
    width=bar_width,
    label="Random Forest"
)

plt.bar(
    x_positions + bar_width / 2,
    tree_importances[forest_indices],
    width=bar_width,
    label="Decision Tree"
)

plt.title(
    "Random Forest vs. Decision Tree Feature Importance"
)

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(
    x_positions,
    data.feature_names[forest_indices],
    rotation=90
)

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. SUMMARIZE THE RESULTS
# ============================================================

forest_top_feature = (
    data.feature_names[
        np.argmax(forest_importances)
    ]
)

forest_top_value = (
    np.max(forest_importances)
)

tree_top_feature = (
    data.feature_names[
        np.argmax(tree_importances)
    ]
)

tree_top_value = (
    np.max(tree_importances)
)

forest_nonzero = int(
    np.count_nonzero(forest_importances)
)

tree_nonzero = int(
    np.count_nonzero(tree_importances)
)

print("\nKEY FINDINGS")
print("-" * 65)

print(
    f"1. Random Forest accuracy: "
    f"{forest_accuracy:.4f}."
)

print(
    f"2. Decision Tree accuracy: "
    f"{tree_accuracy:.4f}."
)

print(
    f"3. The Random Forest's highest-importance feature was "
    f"{forest_top_feature}, with an importance of "
    f"{forest_top_value:.4f}."
)

print(
    f"4. The Decision Tree's highest-importance feature was "
    f"{tree_top_feature}, with an importance of "
    f"{tree_top_value:.4f}."
)

print(
    f"5. The Random Forest assigned nonzero importance to "
    f"{forest_nonzero} of {len(data.feature_names)} features."
)

print(
    f"6. The single Decision Tree assigned nonzero importance to "
    f"{tree_nonzero} of {len(data.feature_names)} features."
)

print(
    "7. The Random Forest aggregates importance across 100 trees, "
    "which spreads feature relevance across multiple independently "
    "constructed decision structures."
)

print(
    "8. The single Decision Tree depends on one fitted hierarchy, "
    "so its importance values can be more concentrated in a smaller "
    "number of selected features."
)

print(
    "\nAssessment 7 Random Forest analysis completed successfully."
)
