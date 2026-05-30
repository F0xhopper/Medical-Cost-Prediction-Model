# %% [markdown]
# # Medical Insurance Cost - Exploratory Data Analysis

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load data
df = pd.read_csv("./data/insurance.csv")
df.head()

# %% Shape and types
print(df.shape)
df.info()

# %% Summary statistics
df.describe()

# %% Missing values
df.isnull().sum()

# %% Distribution of charges (target variable)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["charges"], kde=True, ax=axes[0])
axes[0].set_title("Distribution of Charges")

sns.histplot(np.log1p(df["charges"]), kde=True, ax=axes[1])
axes[1].set_title("Log Distribution of Charges")

plt.tight_layout()
plt.show()

# %% Charges by smoker status
sns.boxplot(data=df, x="smoker", y="charges")
plt.title("Charges by Smoker Status")
plt.show()

# %% Charges by region
sns.boxplot(data=df, x="region", y="charges")
plt.title("Charges by Region")
plt.show()

# %% Age vs charges (coloured by smoker)
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.6)
plt.title("Age vs Charges")
plt.show()

# %% BMI vs charges (coloured by smoker)
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=0.6)
plt.title("BMI vs Charges")
plt.show()

# %% Correlation heatmap
num_df = df.copy()
num_df["sex"] = num_df["sex"].map({"male": 0, "female": 1})
num_df["smoker"] = num_df["smoker"].map({"yes": 1, "no": 0})
num_df = pd.get_dummies(num_df, columns=["region"], drop_first=True)

sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()

# %% Value counts for categorical columns
for col in ["sex", "smoker", "region"]:
    print(f"\n{col}:\n{df[col].value_counts()}")
