import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('academic_performance_dataset_V2.csv')

# Binning CGPA into categories: Low, Medium, High
df['Label'] = pd.cut(df['CGPA'], bins=[0, 2.5, 3.5, 4], labels=['Low', 'Medium', 'High'])

# Drop rows with missing labels
df = df.dropna(subset=['Label'])

# Prepare features and labels
X = df.drop(columns=['ID No', 'CGPA', 'Label'])  # Drop unused
X = pd.get_dummies(X)  # One-hot encode categorical vars
y = df['Label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_labels = le.classes_

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
