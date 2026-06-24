# Name: Isaack Newton Imbwagah
# Title: Explanatory Data Analysis
# Cyber Shujaa ID: CS-DA02 -25030
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
# Load Titanic dataset
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
# Shape of the data set(rows, columns)
print("Dataset shape:", df.shape)
# Column names
print("Columns:", df.columns)
# Info datatypes # non null values
print("Dataset info")
df.info
# Summary statistics for numerical and categorical columns
df.describe(include='all')
# Handle missing values and outliers
print("Missing values per column:", df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Age"])
plt.title("Outlier detection")
plt.show()
# Univariate Analysis - Numerical
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.show()
    # Univariate Analysis - Categorical
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col], palette='pastel')
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()
    plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.title("Target Variable Distribution - Survived")
plt.show()

print("\nSurvival Rate Breakdown:")
print(df['Survived'].value_counts(normalize=True) * 100)
# Numerical vs Target
for col in ['Age', 'Fare']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Survived', y=col, data=df, palette='Set2')
    plt.title(f"{col} vs Survived")
    plt.show()
    # Categorical vs Target
for col in ['Sex', 'Pclass', 'Embarked']:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='Survived', data=df, palette='coolwarm')
    plt.title(f"Survival by {col}")
    plt.show()
    # Encode categorical vars for correlation
df_encoded = df.copy()
df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded = pd.get_dummies(df_encoded, columns=['Embarked'], drop_first=True)

# Correlation Matrix
corr = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

target = 'Survived'
corr_target = corr[target].sort_values(ascending=False)
print("Correlation of features with target variable:\n")
print(corr_target)
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Final Correlation Heatmap")
plt.show()
df.to_csv("titanic_cleaned.csv", index=False)
from IPython.display import FileLink
FileLink(r'titanic_cleaned.csv')
