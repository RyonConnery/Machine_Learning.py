# ============================================================
# Build a Decision Tree
# ============================================================

# Import the libraries required for data handling, model training,
# prediction, accuracy evaluation, tree visualization, and plotting.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD AND EXPLORE THE IRIS DATASET
# ============================================================

# Load the Iris dataset from Scikit-Learn.
iris = load_iris()

# X contains the four flower measurements used as predictors.
X = iris.data

# y contains the species labels:
# 0 = setosa, 1 = versicolor, and 2 = virginica.
y = iris.target

# Store the feature names so they can be displayed in the
# DataFrame, decision tree, and feature-importance results.
feature_names = iris.feature_names

# Convert the feature matrix to a DataFrame for readable inspection.
df = pd.DataFrame(
    X,
    columns=feature_names
)

# Add the target and species names for easier interpretation.
df["target"] = y
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("IRIS DATASET LOADED SUCCESSFULLY")
print("-" * 60)

print(f"Number of observations: {len(df)}")
print(f"Number of predictor features: {len(feature_names)}")

print("\nFirst five rows:")
display(df.head())


# ============================================================
# 2. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Use the exact 80/20 train-test split and random state
# specified in Capella's assessment instructions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING AND TESTING SETS")
print("-" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 3. BUILD THE DECISION TREE CLASSIFIER
# ============================================================

# Create the decision tree using the maximum depth of 3
# specified in Capella's supplemental instructions.
#
# DecisionTreeClassifier uses the Gini impurity criterion by
# default to determine which feature and threshold should be
# selected at each split.
tree_clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Train the classifier using the labeled training observations.
tree_clf.fit(
    X_train,
    y_train
)


# ============================================================
# 4. EVALUATE THE MODEL
# ============================================================

# Predict Iris species for the unseen testing observations.
y_pred = tree_clf.predict(X_test)

# Accuracy measures the proportion of test observations
# that were assigned to the correct species.
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nMODEL EVALUATION")
print("-" * 60)

print(
    f"Decision Tree Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Correct predictions: "
    f"{(y_pred == y_test).sum()} of {len(y_test)}"
)


# ============================================================
# 5. VISUALIZE THE DECISION TREE
# ============================================================

# The plotted tree shows each learned split, Gini impurity,
# sample count, class distribution, and predicted class.
plt.figure(figsize=(14, 9))

plot_tree(
    tree_clf,
    filled=True,
    feature_names=feature_names,
    class_names=iris.target_names,
    rounded=True
)

plt.title(
    "Decision Tree Classifier for the Iris Dataset"
)

plt.tight_layout()
plt.show()


# ============================================================
# 6. EVALUATE FEATURE IMPORTANCE
# ============================================================

# feature_importances_ measures how much each feature contributed
# to reducing impurity across the trained decision tree.
importances = tree_clf.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nFEATURE IMPORTANCE")
print("-" * 60)

display(
    feature_importance_df.style.format({
        "Importance": "{:.4f}"
    })
)

# Print the same values individually, matching the format
# demonstrated in Capella's supplemental instructions.
for name, importance in zip(
    feature_names,
    importances
):
    print(
        f"{name}: "
        f"{importance:.4f}"
    )


# ============================================================
# 7. VISUALIZE FEATURE IMPORTANCE
# ============================================================

# Plot the calculated importance values so the relative
# contribution of each Iris measurement is easy to compare.
plt.figure(figsize=(9, 5))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.title(
    "Decision Tree Feature Importance"
)

plt.xlabel("Iris Feature")
plt.ylabel("Importance")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.tight_layout()
plt.show()


# ============================================================
# 8. SUMMARIZE THE RESULTS
# ============================================================

most_important_feature = (
    feature_importance_df.iloc[0]["Feature"]
)

most_important_value = (
    feature_importance_df.iloc[0]["Importance"]
)

print("\nKEY FINDINGS")
print("-" * 60)

print(
    f"1. The decision tree achieved an accuracy of "
    f"{accuracy:.4f} on the testing set."
)

print(
    f"2. The most important feature was "
    f"{most_important_feature}, with an importance of "
    f"{most_important_value:.4f}."
)

print(
    "3. The trained tree used Gini impurity to select "
    "feature thresholds that increasingly separated the "
    "three Iris species."
)

print(
    "4. Features with greater importance contributed more "
    "to reducing impurity across the learned decision tree."
)

print(
    "5. The maximum depth of 3 limits tree complexity and "
    "keeps the learned decision rules interpretable."
)

print(
    "\nAssessment 6 decision tree analysis completed successfully."
)
