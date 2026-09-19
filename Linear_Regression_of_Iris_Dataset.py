# ============================================================
# Linear Regression Analysis of the Iris Dataset
# ============================================================

# Import the libraries used for data loading, numerical analysis,
# visualization, model training, and evaluation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. LOAD THE IRIS DATASET
# ============================================================

# Load the Iris dataset directly from Scikit-Learn.
iris = load_iris()

# Convert the four measurement features into a pandas DataFrame.
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add the numeric target required by the assessment.
# The three Iris species are represented as:
# 0 = setosa, 1 = versicolor, and 2 = virginica.
df["target"] = iris.target

print("IRIS DATASET LOADED SUCCESSFULLY")
print("-" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nFirst five rows:")
display(df.head())


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

# describe() summarizes the count, mean, standard deviation,
# quartiles, minimum, and maximum for each numerical variable.
print("\nDESCRIPTIVE STATISTICS")
print("-" * 60)
display(df.describe())


# Visualize the distribution of each Iris measurement.
df[iris.feature_names].hist(
    bins=15,
    figsize=(10, 8)
)

plt.suptitle(
    "Distribution of Iris Features",
    fontsize=14
)

plt.tight_layout()
plt.show()


# Plot petal length against the numerical target to visualize
# how this feature changes across the three encoded species values.
plt.figure(figsize=(9, 6))

plt.scatter(
    df["petal length (cm)"],
    df["target"],
    alpha=0.70
)

plt.title("Petal Length and Iris Target")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Target: 0 = Setosa, 1 = Versicolor, 2 = Virginica")
plt.yticks(
    [0, 1, 2],
    ["Setosa (0)", "Versicolor (1)", "Virginica (2)"]
)

plt.tight_layout()
plt.show()


# ============================================================
# 3. PREPARE THE DATA
# ============================================================

# Separate the four flower measurements from the numerical target.
X = df.drop("target", axis=1)
y = df["target"]

# Split the observations into training and testing sets using
# the exact parameters specified in Capella's instructions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING AND TESTING DATA")
print("-" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 4. BUILD THE LINEAR REGRESSION MODEL
# ============================================================

# Create the required ordinary least-squares linear regression model.
model = LinearRegression()

# Fit the model using the training features and numerical target.
model.fit(
    X_train,
    y_train
)


# ============================================================
# 5. EVALUATE THE MODEL
# ============================================================

# Calculate R-squared on the unseen testing data.
# R-squared measures how much of the variation in the numerical
# target is explained by the fitted linear model.
r_squared = model.score(
    X_test,
    y_test
)

# Predict target values for the testing observations.
y_pred = model.predict(X_test)

print("\nMODEL EVALUATION")
print("-" * 60)

print(f"R-squared: {r_squared:.4f}")


# Create a table so the predicted values can be compared directly
# with the actual numerical species targets.
prediction_comparison = pd.DataFrame({
    "Actual Target": y_test.to_numpy(),
    "Predicted Target": y_pred
})

prediction_comparison["Prediction Error"] = (
    prediction_comparison["Actual Target"]
    - prediction_comparison["Predicted Target"]
)

print("\nActual vs. Predicted Target Values:")
display(
    prediction_comparison.round(4)
)


# Visualize actual and predicted values.
# Points closer to the diagonal reference line represent predictions
# that are closer to the actual numerical target values.
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.75
)

plt.plot(
    [0, 2],
    [0, 2],
    linestyle="--"
)

plt.title("Actual vs. Predicted Iris Target Values")
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")

plt.xticks(
    [0, 1, 2],
    ["Setosa (0)", "Versicolor (1)", "Virginica (2)"]
)

plt.tight_layout()
plt.show()


# ============================================================
# 6. INTERPRET THE MODEL COEFFICIENTS
# ============================================================

# Each coefficient represents the estimated change in the predicted
# numerical target associated with a one-unit increase in that feature,
# while the other features are held constant.
coefficient_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

# Add the absolute coefficient magnitude to make relative coefficient
# sizes easier to compare.
coefficient_table["Absolute Coefficient"] = (
    coefficient_table["Coefficient"].abs()
)

coefficient_table = coefficient_table.sort_values(
    by="Absolute Coefficient",
    ascending=False
).reset_index(drop=True)

print("\nMODEL COEFFICIENTS")
print("-" * 60)

display(
    coefficient_table.round(4)
)

print(f"Intercept: {model.intercept_:.4f}")


# ============================================================
# 7. COEFFICIENT INTERPRETATION
# ============================================================

print("\nCOEFFICIENT INTERPRETATION")
print("-" * 60)

for _, row in coefficient_table.iterrows():

    feature = row["Feature"]
    coefficient = row["Coefficient"]

    if coefficient > 0:
        direction = "positive"
    elif coefficient < 0:
        direction = "negative"
    else:
        direction = "no linear"

    print(
        f"{feature}: coefficient = {coefficient:.4f}. "
        f"This indicates a {direction} relationship with the "
        f"numerically encoded Iris target when the other features "
        f"are held constant."
    )


# ============================================================
# 8. KEY FINDINGS
# ============================================================

largest_feature = coefficient_table.iloc[0]["Feature"]
largest_coefficient = coefficient_table.iloc[0]["Coefficient"]

print("\nKEY FINDINGS")
print("-" * 60)

print(
    f"1. The linear regression model achieved an R-squared "
    f"value of {r_squared:.4f} on the testing data."
)

print(
    "2. The predicted values generally follow the numerical "
    "ordering of the three Iris species targets."
)

print(
    f"3. The feature with the largest coefficient magnitude is "
    f"{largest_feature}, with a coefficient of "
    f"{largest_coefficient:.4f}."
)

print(
    "4. Positive coefficients increase the model's predicted "
    "numerical target, while negative coefficients decrease it "
    "when the other features are held constant."
)

print(
    "5. Coefficient magnitude can indicate stronger influence "
    "within this fitted model, but magnitude must be interpreted "
    "carefully because the features use different measurement scales."
)

print(
    "6. The Iris target values represent encoded species categories, "
    "so this regression model is being used for the specific "
    "instructional purpose required by the assessment."
)

print(
    "\nAssessment 4 linear regression analysis completed successfully."
)
