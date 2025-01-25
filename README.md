# Predictive Sampling for Credit Card Fraud Detection

This project implements a credit card fraud detection system using machine learning. It tackles the class imbalance problem using Synthetic Minority Oversampling Technique (SMOTE) and evaluates multiple sampling techniques to test the performance of various machine learning models.

## Project Workflow

### Step 1: Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

### Step 2: Load and Clean Dataset
- *Dataset*: The dataset used in this project is Creditcard_data.csv.
- *Data Cleaning*: Replace any erroneous or placeholder values.
```python
# Load the dataset
data = pd.read_csv('Creditcard_data.csv')

# Replace erroneous data (example of data cleaning)
data = data.replace('lakshit gupta', 'lakshit gupta')
```


### Step 3: Handle Class Imbalance Using SMOTE
- Separate the dataset into features and target variable.
- Apply SMOTE to generate a balanced dataset.
```python
# Split the dataset into features and target variable
y = data['Class']
x = data.drop('Class', axis=1)

# Perform SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x, y)

# Combine the resampled data into a single DataFrame
balanced_data = pd.concat([
    pd.DataFrame(x_smote, columns=x.columns),
    pd.DataFrame(y_smote, columns=['Class'])
], axis=1)
```

### Step 4: Generate Samples Using Different Strategies
1. *Random Sampling*: Select a random fraction of the data.
2. *Stratified Sampling*: Ensure the same class distribution in the sample as the original dataset.
3. *Systematic Sampling*: Select samples at regular intervals.
4. *Cluster Sampling*: Divide data into clusters and sample one cluster.
5. *Bootstrap Sampling*: Generate a sample with replacement.
```python
# Random Sampling
sample1 = balanced_data.sample(frac=0.2, random_state=42)

# Stratified Sampling
sample2 = balanced_data.groupby('Class', group_keys=False).apply(
    lambda grp: grp.sample(frac=0.2, random_state=42)
)

# Systematic Sampling
k = len(balanced_data) // int(0.2 * len(balanced_data))
start = np.random.randint(0, k)
sample3 = balanced_data.iloc[start::k]

# Cluster Sampling
num_clusters = 5
balanced_data['Cluster'] = np.arange(len(balanced_data)) % num_clusters
selected_cluster = np.random.choice(num_clusters)
sample4 = balanced_data[balanced_data['Cluster'] == selected_cluster].drop('Cluster', axis=1)

# Bootstrap Sampling
sample5 = balanced_data.sample(n=int(0.2 * len(balanced_data)), replace=True, random_state=42)
```

### Step 5: Define and Train Machine Learning Models
- Models used:
  - Logistic Regression
  - Decision Tree
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (k-NN)
- Evaluate the performance of each model on different samples.
```python
# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier()
}

# Initialize results storage
results = {}
samples = [sample1, sample2, sample3, sample4, sample5]

# Evaluate models on each sample
for model_name, model in models.items():
    results[model_name] = []
    for sample in samples:
        X_sample = sample.drop('Class', axis=1)
        y_sample = sample['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name].append(accuracy)
```

### Step 6: Save and Display Results
- Store model accuracies in a CSV file.
- Print results for quick reference.
```python
# Save results to a CSV file
results_df = pd.DataFrame(results, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
results_df.to_csv("model_accuracies.csv")

# Print results for quick reference
print("Model accuracies saved to 'model_accuracies.csv'")
```

## Results Summary
- Each sample was evaluated using the five machine learning models.
- The results were saved in a CSV file named model_accuracies.csv.

### Key Points:
1. *Random Sampling*: Provides a general overview of the model's performance.
2. *Stratified Sampling*: Ensures balanced representation of classes in the sample.
3. *Systematic Sampling*: Useful for evenly distributed datasets.
4. *Cluster Sampling*: Evaluates models on a specific subset (cluster) of the dataset.
5. *Bootstrap Sampling*: Tests model robustness with repetitive sampling.

## How to Run the Code
1. Ensure the dataset Creditcard_data.csv is in the working directory.
2. Install necessary libraries if not already installed:
  ``` bash
   pip install pandas numpy matplotlib scikit-learn imbalanced-learn
   ```
3. Run the script to generate model_accuracies.csv and print the results.

## Conclusion
This project provides insights into credit card fraud detection using different sampling techniques and machine learning models. The workflow ensures that class imbalance is addressed, and models are evaluated thoroughly across various sampling strategies.
