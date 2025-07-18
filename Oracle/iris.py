import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Take 100 random rows
df_sample = df.sample(n=100, random_state=42).reset_index(drop=True)

# Add Id column
df_sample.insert(0, 'Id', range(1, 101))

# Save to CSV
df_sample.to_csv("iris.csv", index=False)
