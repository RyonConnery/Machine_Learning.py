# ============================================================
# Assessment 2: Data Exploration and Visualization
# Housing Prices Data Exploration
# ============================================================

# Import the libraries required for data handling, numerical analysis,
# statistical exploration, and visualization.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown


# ============================================================
# 1. LOAD THE CALIFORNIA HOUSING DATASET
# ============================================================

# Load the California housing dataset from the public source that matches
# the dataset used in the course resource.
data_url = (
    "https://raw.githubusercontent.com/"
    "parmarshashank/California-Housing-Prices/"
    "master/datasets/housing/housing.csv"
)

df = pd.read_csv(data_url)

display(Markdown("## 1. Dataset Loaded Successfully"))

# Display the first five records so the dataset structure can be verified.
print("First five rows of the dataset:")
display(df.head())

print("\nDataset shape:")
print(f"{df.shape[0]:,} rows and {df.shape[1]} columns")

print("\nColumn data types and non-null counts:")
df.info()


# ============================================================
# 2. IDENTIFY AND HANDLE MISSING VALUES
# ============================================================

display(Markdown("## 2. Data Cleaning"))

# Count missing values in every column to determine which features require
# cleaning before exploratory analysis.
missing_before = df.isnull().sum()

print("Missing values before cleaning:")
print(missing_before)

# total_bedrooms contains missing observations. The median is used for
# imputation because it is less sensitive than the mean to extreme values
# in a right-skewed housing-count variable.
bedroom_median = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(bedroom_median)

print(
    f"\nMissing total_bedrooms values were replaced with the median: "
    f"{bedroom_median:.1f}"
)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 3. CHECK FOR DUPLICATE RECORDS
# ============================================================

# Duplicate rows can unintentionally give repeated districts more influence
# in an analysis, so exact duplicates are identified and removed if present.
duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows detected: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {duplicate_count}")
else:
    print("No duplicate rows required removal.")


# ============================================================
# 4. DETECT AND ADDRESS NUMERICAL OUTLIERS
# ============================================================

display(Markdown("## 3. Outlier Detection and Treatment"))

# The IQR method identifies unusually extreme observations without assuming
# that the features follow a normal distribution.
#
# Outlier treatment is limited to heavily skewed predictor variables whose
# extreme magnitudes can dominate later analyses. Geographic coordinates,
# housing age, and the target variable are retained because their extreme
# values can represent meaningful geographic or market information.
outlier_columns = [
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

outlier_summary = []

for column in outlier_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())

    outlier_summary.append({
        "Feature": column,
        "Outliers Detected": outlier_count,
        "Lower IQR Bound": round(lower_bound, 2),
        "Upper IQR Bound": round(upper_bound, 2)
    })

    # Cap extreme predictor values at the IQR boundaries rather than deleting
    # entire districts. This preserves the records while limiting the influence
    # of unusually large or small predictor values.
    df[column] = df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

outlier_summary_df = pd.DataFrame(outlier_summary)

print("IQR outlier summary:")
display(outlier_summary_df)

print(
    "\nOutlier treatment completed by capping selected predictor values "
    "at their IQR boundaries."
)


# ============================================================
# 5. GENERATE DESCRIPTIVE STATISTICS
# ============================================================

display(Markdown("## 4. Descriptive Statistics"))

# describe() summarizes the central tendency, spread, quartiles, and ranges
# of the numerical variables after cleaning.
summary_statistics = df.describe()

print("Summary statistics:")
display(summary_statistics)


# ============================================================
# 6. VISUALIZE NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

display(Markdown("## 5. Numerical Feature Distributions"))

# Histograms show the shape, concentration, and skewness of each numerical
# variable in the cleaned dataset.
numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns].hist(
    bins=40,
    figsize=(16, 12)
)

plt.suptitle(
    "Distributions of Numerical Features",
    fontsize=16,
    y=1.02
)

plt.tight_layout()
plt.show()


# A box plot provides another view of the housing-price distribution and
# highlights the spread of median house values.
plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["median_house_value"]
)

plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value ($)")
plt.tight_layout()
plt.show()


# ============================================================
# 7. VISUALIZE THE CATEGORICAL FEATURE
# ============================================================

display(Markdown("## 6. Ocean Proximity Distribution"))

# ocean_proximity is the dataset's categorical feature. A count plot shows
# how frequently districts occur in each location category.
plt.figure(figsize=(10, 6))

category_order = df["ocean_proximity"].value_counts().index

sns.countplot(
    data=df,
    x="ocean_proximity",
    order=category_order
)

plt.title("Number of Districts by Ocean Proximity")
plt.xlabel("Ocean Proximity")
plt.ylabel("District Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

print("Ocean proximity category counts:")
print(df["ocean_proximity"].value_counts())


# ============================================================
# 8. EXPLORE RELATIONSHIPS WITH HOUSING PRICES
# ============================================================

display(Markdown("## 7. Relationships With Housing Prices"))

# Median income is examined against median house value because income is an
# economically meaningful predictor and is strongly related to housing prices
# in the course dataset.
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="median_income",
    y="median_house_value",
    alpha=0.30
)

plt.title("Median Income vs. Median House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value ($)")
plt.tight_layout()
plt.show()


# Geographic coordinates are also examined because California housing prices
# vary substantially by location.
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["median_house_value"],
    s=10,
    alpha=0.35
)

plt.colorbar(
    scatter,
    label="Median House Value ($)"
)

plt.title("Geographic Distribution of California Housing Prices")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()


# Compare housing prices across ocean-proximity categories to determine
# whether coastal location is associated with differences in house values.
plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df,
    x="ocean_proximity",
    y="median_house_value",
    order=category_order
)

plt.title("Median House Value by Ocean Proximity")
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value ($)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ============================================================
# 9. CALCULATE CORRELATIONS
# ============================================================

display(Markdown("## 8. Correlation Analysis"))

# Pearson correlations are calculated for numerical features only because
# ocean_proximity is categorical.
correlation_matrix = df[numeric_columns].corr()

print("Correlation matrix:")
display(correlation_matrix.round(3))

# Display correlations with the target from strongest positive relationship
# to strongest negative relationship.
house_value_correlations = (
    correlation_matrix["median_house_value"]
    .sort_values(ascending=False)
)

print("\nCorrelations with median_house_value:")
print(house_value_correlations.round(3))


# ============================================================
# 10. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    square=True
)

plt.title("Correlation Heatmap of Numerical Housing Features")
plt.tight_layout()
plt.show()


# ============================================================
# 11. SUMMARIZE KEY EXPLORATORY RESULTS
# ============================================================

display(Markdown("## 9. Exploratory Analysis Summary"))

strongest_feature = (
    house_value_correlations
    .drop("median_house_value")
    .abs()
    .idxmax()
)

strongest_value = house_value_correlations[strongest_feature]

print("KEY FINDINGS")
print("-" * 60)

print(
    f"1. The cleaned dataset contains "
    f"{df.shape[0]:,} rows and {df.shape[1]} columns."
)

print(
    "2. Missing total_bedrooms observations were imputed "
    f"with the median value of {bedroom_median:.1f}."
)

print(
    f"3. Exact duplicate rows detected before cleaning: "
    f"{duplicate_count}."
)

print(
    f"4. The numerical feature with the strongest linear "
    f"relationship to median_house_value is "
    f"{strongest_feature}, with a correlation of "
    f"{strongest_value:.3f}."
)

print(
    "5. The geographic and ocean-proximity visualizations show "
    "that housing value varies substantially by location."
)

print(
    "6. Several count-based predictors have skewed distributions, "
    "which should be considered during feature engineering and "
    "future model selection."
)

print(
    "7. The exploration indicates that income and location should "
    "receive particular attention in later housing-price prediction models."
)


# ============================================================
# 12. MACHINE LEARNING IMPLICATIONS
# ============================================================

display(Markdown("## 10. Implications for Future Machine Learning"))

print(
    """
Future modeling should evaluate median_income and geographic/location
information carefully because the exploratory analysis shows meaningful
relationships with median_house_value. Categorical ocean_proximity will
require numerical encoding before most machine learning algorithms can use it.

The skewed distributions of several housing-count variables may justify
scaling, transformation, or engineered ratio features in later assessments.
The capped upper values visible in some housing variables should also be
considered because artificial limits in the source data can affect model
learning and evaluation.

Three practical machine learning use cases for this work are:
1. Predicting district-level median housing values for real-estate analysis.
2. Identifying geographic areas with characteristics associated with higher
   or lower housing values for investment research.
3. Supporting housing-market forecasting and planning by analyzing how
   income, location, population, and housing characteristics relate to price.
"""
)

print("\nAssessment 2 exploratory data analysis completed successfully.")
