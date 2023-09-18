import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("../data/balanced_nonrandom_data.csv")

# Split the dataset into features and target variable
X = data.drop('accident binary', axis=1)
y = data['accident binary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Apply PCA to the normalized data
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define the XGBoost model
model = xgb.XGBClassifier()

# Train the old model on the pre-processed training data
# model.fit(X_train, y_train)

# Evaluate the old model on the pre-processed testing data
# y_pred = model.predict(X_test)

# Train the new model using PCA training data
model.fit(X_train_pca, y_train)

# Evaluate the new model on the PCA testing data
y_pred = model.predict(X_test_pca)
print("Metrics for XGBoost model with balanced_nonrandom_data.csv:")
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))