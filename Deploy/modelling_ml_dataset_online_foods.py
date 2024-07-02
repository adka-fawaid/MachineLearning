import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("onlinefoods.csv")

# Drop unnecessary columns
data = data.drop(['Unnamed: 12','Pin code', 'latitude', 'longitude'], axis=1)

# Separate features (x) and target (y)
x = data.drop(['Feedback'], axis=1)
y = data['Feedback']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define preprocessing steps using Pipeline
preprocessor = Pipeline([
    ('encoder', LabelEncoder()),  # Label encoding for categorical columns
    ('scaler', StandardScaler())  # Standard scaling for numerical columns
])

# Fit and transform on training data
x_train_prep = preprocessor.fit_transform(x_train)
# Transform on testing data (without fitting to prevent data leakage)
x_test_prep = preprocessor.transform(x_test)

# Initialize and train SVM model
clf = SVC()
clf.fit(x_train_prep, y_train)

# Predict on testing data
y_pred = clf.predict(x_test_prep)

# Evaluate model
clf_acc = accuracy_score(y_test, y_pred)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy SVM: {:.2f}%".format(clf_acc * 100))
