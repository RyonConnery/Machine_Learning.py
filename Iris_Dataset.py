# ============================================================
# Iris Dataset Classifier
# ============================================================

# Import the libraries required for numerical operations, data handling,
# visualization, model training, and model evaluation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# 1. LOAD THE IRIS DATASET
# ============================================================

# Load the built-in Iris dataset from Scikit-Learn.
iris = datasets.load_iris()

# Convert the feature data into a pandas DataFrame so the dataset
# can be explored and visualized more easily.
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add the numeric species target to the DataFrame.
df["target"] = iris.target

# Add readable species names for easier interpretation.
df["species"] = df["target"].map(
    {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
)

print("IRIS DATASET LOADED SUCCESSFULLY")
print("-" * 50)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nFirst five rows:")
display(df.head())


# ============================================================
# 2. EXPLORE THE DATASET
# ============================================================

# describe() provides descriptive statistics such as the mean,
# standard deviation, quartiles, minimum, and maximum.
print("\nDESCRIPTIVE STATISTICS")
print("-" * 50)
display(df.describe())

# Show the number of observations in each species class.
print("\nSPECIES COUNTS")
print("-" * 50)
print(df["species"].value_counts())


# ============================================================
# 3. VISUALIZE THE DATA
# ============================================================

# Plot petal length against petal width because these features
# provide strong visual separation among the three Iris species.
plt.figure(figsize=(9, 6))

species_colors = {
    "setosa": "tab:blue",
    "versicolor": "tab:orange",
    "virginica": "tab:green"
}

for species_name in df["species"].unique():
    species_data = df[df["species"] == species_name]

    plt.scatter(
        species_data["petal length (cm)"],
        species_data["petal width (cm)"],
        label=species_name,
        alpha=0.75
    )

plt.title("Iris Species by Petal Length and Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.show()


# Plot feature distributions using histograms.
df[iris.feature_names].hist(
    bins=15,
    figsize=(10, 8)
)

plt.suptitle(
    "Distribution of Iris Numerical Features",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ============================================================
# 4. PREPROCESS THE DATA
# ============================================================

# Separate the four measurement features from the target labels.
X = iris.data
y = iris.target

# Split the dataset into training and testing sets.
# Stratification preserves the class proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING AND TESTING SPLIT")
print("-" * 50)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 5. BUILD THE CLASSIFIER
# ============================================================

# Logistic Regression is used because this is a supervised
# multiclass classification problem with four numerical features.
classifier = LogisticRegression(
    max_iter=200,
    random_state=42
)

# Train the classifier using the labeled training data.
classifier.fit(
    X_train,
    y_train
)


# ============================================================
# 6. TEST THE CLASSIFIER
# ============================================================

# Predict the species labels for the unseen testing data.
y_pred = classifier.predict(X_test)

# Accuracy measures the proportion of correct predictions.
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nMODEL PERFORMANCE")
print("-" * 50)
print(
    f"Logistic Regression Accuracy: "
    f"{accuracy:.4f}"
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

# The confusion matrix shows the number of correct and incorrect
# predictions for each Iris species.
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

display_cm = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display_cm.plot()
plt.title("Iris Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

# The classification report provides precision, recall, and F1-score
# for each species, offering more detail than accuracy alone.
print("\nCLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ============================================================
# 9. INTERPRET THE RESULTS
# ============================================================

print("\nKEY FINDINGS")
print("-" * 50)

print(
    f"1. The Logistic Regression classifier achieved an "
    f"accuracy of {accuracy:.4f} on the test set."
)

print(
    "2. Petal length and petal width provide strong visual "
    "separation among the Iris species."
)

print(
    "3. Setosa is especially distinct from the other species, "
    "while versicolor and virginica have more overlap."
)

print(
    "4. The confusion matrix shows exactly which species were "
    "classified correctly and where any misclassifications occurred."
)

print(
    "5. The model demonstrates a complete supervised machine "
    "learning classification workflow using structured numerical data."
)

print("\nAssessment 3 Iris classification completed successfully.")
