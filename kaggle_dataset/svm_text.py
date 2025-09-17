import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('academic_performance_dataset_V2.csv')

# Create label based on CGPA bins
df['Label'] = pd.cut(df['CGPA'], bins=[0, 2.5, 3.5, 4], labels=['Low', 'Medium', 'High'])

# Drop rows with missing labels
df = df.dropna(subset=['Label'])

# Features and target
X = df.drop(columns=['ID No', 'CGPA', 'Label'])  # Drop ID and CGPA
X = pd.get_dummies(X)  # One-hot encoding for categorical features
y = df['Label']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# SVM Model with linear kernel
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\n=== Classification Report (SVM) ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_labels = le.classes_

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
