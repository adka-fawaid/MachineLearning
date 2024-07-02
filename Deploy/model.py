import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

# Load dataset
data = pd.read_csv("onlinefoods.csv")

# Drop unnecessary columns
data = data.drop(['Unnamed: 12','Pin code', 'latitude', 'longitude'], axis=1)

# Separate features (x) and target (y)
x = data.drop(['Feedback'], axis=1)
y = data['Feedback']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocessing function to encode categorical variables
def preprocess_data(x):
    le_dict = {}
    for column in x.columns:
        if x[column].dtype == 'object':
            le = LabelEncoder()
            x[column] = le.fit_transform(x[column])
            le_dict[column] = le
    return x, le_dict

# Preprocess training and testing data
x_train, le_dict = preprocess_data(x_train)
x_test, _ = preprocess_data(x_test)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize and train SVM model
clf = SVC()
clf.fit(x_train, y_train)

# Evaluate model
y_pred = clf.predict(x_test)
clf_acc = accuracy_score(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)
clf_conf_matrix = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(clf_report)
print("Confusion Matrix:")
print(clf_conf_matrix)
print("Accuracy SVM: {:.2f}%".format(clf_acc * 100))

# Save the preprocessing pipeline and model
with open('scaler.sav', 'wb') as file:
    pickle.dump(scaler, file)

with open('label_encoders.sav', 'wb') as file:
    pickle.dump(le_dict, file)

with open('svm_model.sav', 'wb') as file:
    pickle.dump(clf, file)

# Save the columns for verification
with open('columns.sav', 'wb') as file:
    pickle.dump(x.columns, file)

# Save the evaluation metrics
with open('metrics.sav', 'wb') as file:
    pickle.dump({
        'accuracy': clf_acc,
        'classification_report': clf_report,
        'confusion_matrix': clf_conf_matrix
    }, file)
