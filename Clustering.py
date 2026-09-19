# ============================================================
# Clustering Techniques
# ============================================================

# Import the libraries required for file upload, data preparation,
# preprocessing, K-means clustering, classification, evaluation,
# and visualization.
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. UPLOAD AND LOAD THE CREDIT CARD CUSTOMERS DATASET
# ============================================================

# Upload the CSV directly into the current Colab runtime.
# This avoids relying on a filename that may not exist in the
# temporary notebook environment.
print("Select the Credit Card Customers CSV file.")

uploaded = files.upload()

# Identify the uploaded CSV automatically.
csv_files = [
    filename
    for filename in uploaded.keys()
    if filename.lower().endswith(".csv")
]

if len(csv_files) != 1:
    raise ValueError(
        "Please upload exactly one CSV file for Assessment 9."
    )

csv_filename = csv_files[0]

# Read the uploaded CSV directly from memory.
data = pd.read_csv(
    io.BytesIO(uploaded[csv_filename])
)

print("\nCREDIT CARD CUSTOMERS DATASET LOADED SUCCESSFULLY")
print("-" * 70)

print(f"Uploaded file: {csv_filename}")
print(f"Number of observations: {data.shape[0]:,}")
print(f"Number of original columns: {data.shape[1]}")

print("\nFirst five rows:")
display(data.head())


# ============================================================
# 2. PREPARE THE ACTUAL CHURN TARGET
# ============================================================

# The supplied dataset uses Attrition_Flag as its churn variable.
# Encode:
# Existing Customer = 0
# Attrited Customer = 1
y = data["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

# Confirm that every target value was encoded successfully.
if y.isna().any():
    raise ValueError(
        "Unexpected values were found in Attrition_Flag."
    )

print("\nTARGET DISTRIBUTION")
print("-" * 70)

print(data["Attrition_Flag"].value_counts())


# ============================================================
# 3. REMOVE IDENTIFIER AND TARGET-LEAKAGE COLUMNS
# ============================================================

# CLIENTNUM is an identifier rather than a behavioral predictor.
#
# The two Naive_Bayes_Classifier columns contain information
# derived from the attrition outcome and would leak target
# information into the churn model.
leakage_columns = [
    column
    for column in data.columns
    if column.startswith("Naive_Bayes_Classifier")
]

columns_to_remove = [
    "Attrition_Flag",
    "CLIENTNUM"
] + leakage_columns

X = data.drop(
    columns=columns_to_remove
)

print("\nPREDICTOR PREPARATION")
print("-" * 70)

print(f"Predictor columns retained: {X.shape[1]}")
print(f"Naive Bayes leakage columns removed: {len(leakage_columns)}")


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

# Numerical features require scaling before K-means because
# distance calculations are affected by differences in scale.
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

# Text-based categorical predictors must be converted into
# numerical indicator variables.
categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nFEATURE TYPES")
print("-" * 70)

print(f"Numerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

print("\nCategorical columns:")
for column in categorical_features:
    print(f"- {column}")


# ============================================================
# 5. PREPROCESS THE DATA
# ============================================================

# StandardScaler places numerical variables on comparable scales.
# OneHotEncoder converts categorical values into numeric features.
preprocess = make_column_transformer(
    (
        StandardScaler(),
        numeric_features
    ),
    (
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_features
    )
)

X_processed = preprocess.fit_transform(X)

print("\nPREPROCESSING COMPLETED")
print("-" * 70)

print(f"Processed feature count: {X_processed.shape[1]}")


# ============================================================
# 6. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ============================================================

# Capella specifies a 70/30 train-test split with random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.30,
    random_state=42
)

print("\nTRAINING AND TESTING SETS")
print("-" * 70)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")


# ============================================================
# 7. APPLY K-MEANS CLUSTERING
# ============================================================

# Capella requires four customer clusters.
# K-means groups customers with similar processed attributes.
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(X_train)

# Append each customer's cluster assignment as one new
# engineered feature for the second Random Forest model.
X_train_with_clusters = np.c_[
    X_train,
    kmeans.labels_
]

X_test_with_clusters = np.c_[
    X_test,
    kmeans.predict(X_test)
]

print("\nK-MEANS CLUSTERING")
print("-" * 70)

unique_clusters, cluster_counts = np.unique(
    kmeans.labels_,
    return_counts=True
)

for cluster, count in zip(
    unique_clusters,
    cluster_counts
):
    print(
        f"Cluster {cluster}: "
        f"{count:,} training customers"
    )


# ============================================================
# 8. RANDOM FOREST WITHOUT CLUSTER FEATURE
# ============================================================

# Train the baseline churn classifier without the engineered
# K-means cluster label.
rf = RandomForestClassifier(
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

preds = rf.predict(
    X_test
)

accuracy_without_clusters = accuracy_score(
    y_test,
    preds
)

print("\nMODEL WITHOUT CLUSTER FEATURE")
print("-" * 70)

print(
    f"Accuracy without clusters: "
    f"{accuracy_without_clusters:.4f}"
)


# ============================================================
# 9. RANDOM FOREST WITH CLUSTER FEATURE
# ============================================================

# Train the same classifier after adding the K-means cluster
# assignment as an engineered predictor.
rf_with_clusters = RandomForestClassifier(
    random_state=42
)

rf_with_clusters.fit(
    X_train_with_clusters,
    y_train
)

preds_with_clusters = rf_with_clusters.predict(
    X_test_with_clusters
)

accuracy_with_clusters = accuracy_score(
    y_test,
    preds_with_clusters
)

print("\nMODEL WITH CLUSTER FEATURE")
print("-" * 70)

print(
    f"Accuracy with clusters: "
    f"{accuracy_with_clusters:.4f}"
)


# ============================================================
# 10. COMPARE MODEL ACCURACY
# ============================================================

accuracy_difference = (
    accuracy_with_clusters
    - accuracy_without_clusters
)

comparison_df = pd.DataFrame({
    "Model": [
        "Random Forest Without Cluster Feature",
        "Random Forest With Cluster Feature"
    ],
    "Accuracy": [
        accuracy_without_clusters,
        accuracy_with_clusters
    ]
})

print("\nMODEL ACCURACY COMPARISON")
print("-" * 70)

display(
    comparison_df.style.format({
        "Accuracy": "{:.4f}"
    })
)

print(
    f"Accuracy difference with cluster feature: "
    f"{accuracy_difference:.4f}"
)


# ============================================================
# 11. CLASSIFICATION REPORT WITHOUT CLUSTERS
# ============================================================

print("\nCLASSIFICATION REPORT WITHOUT CLUSTERS")
print("-" * 70)

print(
    classification_report(
        y_test,
        preds,
        target_names=[
            "Existing Customer",
            "Attrited Customer"
        ],
        digits=4
    )
)


# ============================================================
# 12. CLASSIFICATION REPORT WITH CLUSTERS
# ============================================================

print("\nCLASSIFICATION REPORT WITH CLUSTERS")
print("-" * 70)

print(
    classification_report(
        y_test,
        preds_with_clusters,
        target_names=[
            "Existing Customer",
            "Attrited Customer"
        ],
        digits=4
    )
)


# ============================================================
# 13. VISUALIZE THE ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    [
        "Without\nCluster Feature",
        "With\nCluster Feature"
    ],
    [
        accuracy_without_clusters,
        accuracy_with_clusters
    ]
)

plt.title(
    "Customer Churn Prediction Accuracy"
)

plt.ylabel("Accuracy")
plt.ylim(0, 1.0)

plt.tight_layout()
plt.show()


# ============================================================
# 14. FINAL INTERPRETATION
# ============================================================

print("\nKEY FINDINGS")
print("-" * 70)

print(
    f"1. K-means divided the training customers into "
    f"{len(unique_clusters)} clusters."
)

print(
    f"2. Random Forest accuracy without the cluster feature "
    f"was {accuracy_without_clusters:.4f}."
)

print(
    f"3. Random Forest accuracy with the cluster feature "
    f"was {accuracy_with_clusters:.4f}."
)

print(
    f"4. The accuracy difference after adding the customer "
    f"cluster feature was {accuracy_difference:.4f}."
)

if accuracy_difference > 0:
    print(
        "5. The engineered cluster feature improved predictive "
        "accuracy in this experiment."
    )

elif accuracy_difference < 0:
    print(
        "5. The engineered cluster feature slightly reduced "
        "overall accuracy in this experiment, demonstrating that "
        "clustering does not automatically improve a classifier."
    )

else:
    print(
        "5. The engineered cluster feature did not change "
        "overall accuracy in this experiment."
    )

print(
    "6. The classification reports provide precision, recall, "
    "and F1-score results for existing and attrited customers."
)

print(
    "7. K-means acts as an unsupervised feature-engineering "
    "method by representing latent customer group membership "
    "as an additional predictor."
)

print(
    "\nAssessment 9 clustering analysis completed successfully."
)
