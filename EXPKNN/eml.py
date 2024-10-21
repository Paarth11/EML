# Importing Necessary Libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Loading the Dataset from the CSV file
df = pd.read_csv('iris_dataset.csv')

# Preprocessing the Data
X = df.iloc[:, :-1].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying k-Nearest Neighbour
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluating the Model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Printing Correct and Wrong Predictions
correct_predictions = [ ]
wrong_predictions = [ ]

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct_predictions.append((X_test[i], y_test[i], y_pred[i]))
    else:
        wrong_predictions.append((X_test[i], y_test[i], y_pred[i]))

print("\nCorrect Predictions:")
for cp in correct_predictions:
    print(f"Data: {cp[0]}, True Label: {cp[1]}, Predicted Label: {cp[2]}")

print("\nWrong Predictions:")
for wp in wrong_predictions:
    print(f"Data: {wp[0]}, True Label: {wp[1]}, Predicted Label: {wp[2]}")
