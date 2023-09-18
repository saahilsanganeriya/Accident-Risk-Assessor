import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

filepath = r'C:\Users\saahi\Documents\Saahil\college\spring23\cs4641\CS-4641-Project\data\balanced_nonrandom_data.csv'
data = pd.read_csv(filepath)

y = data['accident binary']
X = data.drop('accident binary', axis=1)  # assuming the target column is named 'accident binary'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)
print("Metrics for Random Forest model with balanced_nonrandom_data.csv:")
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))