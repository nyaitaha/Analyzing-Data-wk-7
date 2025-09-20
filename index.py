# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -----------------------------
# Task 1: Load and Explore the Dataset
# -----------------------------

try:
    # Load the Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame  # Convert to pandas DataFrame
    
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the dataset path.")
except Exception as e:
    print(f"❌ Error while loading dataset: {e}")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Explore structure (data types, missing values)
print("\nDataset Information:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset (drop missing values if any)
df = df.dropna()

# -----------------------------
# Task 2: Basic Data Analysis
# -----------------------------

# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping: average petal length by species
avg_petal_length = df.groupby("target")["petal length (cm)"].mean()
print("\nAverage Petal Length per Species:")
print(avg_petal_length)

# -----------------------------
# Task 3: Data Visualization
# -----------------------------
sns.set(style="whitegrid")  # better style

# 1. Line Chart - Petal length trend over sample index
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal length (cm)"], label="Petal Length", color="blue")
plt.title("Line Chart: Petal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(8,5))
avg_petal_length.plot(kind="bar", color=["green","orange","purple"])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=Setosa, 1=Versicolor, 2=Virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram - Distribution of Sepal Length
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=20, color="lightblue", edgecolor="black")
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal length vs Petal length
plt.figure(figsize=(8,5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"],
            c=df["target"], cmap="viridis", alpha=0.7)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()

# -----------------------------
# Findings / Observations
# -----------------------------
print("\n📊 Observations:")
print("- Setosa (species 0) has the smallest petal length, Virginica (species 2) the largest.")
print("- Virginica’s petal length average is significantly higher than the other species.")
print("- Sepal lengths are normally distributed around 5–6 cm.")
print("- There is a clear positive correlation between sepal length and petal length.")
