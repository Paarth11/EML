import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the iris dataset from CSV
df = pd.read_csv('iris_dataset.csv')

# Extract features and target variable
X = df[["petal_length", "petal_width"]].values
y = df["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2}).values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Train an SVM with a linear kernel
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_std, y_train)

# Predictions
y_pred = svm.predict(X_test_std)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
