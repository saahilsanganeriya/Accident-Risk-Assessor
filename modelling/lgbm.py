import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

filepath = r'C:\Users\saahi\Documents\Saahil\college\spring23\cs4641\CS-4641-Project\data\balanced_nonrandom_data.csv'
data = pd.read_csv(filepath)

print('1')

y = data['accident binary']
X = data.drop('accident binary', axis=1)  # assuming the target column is named 'accident binary'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the LightGBM Classifier
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

lgb_classifier = lgb.train(params, train_data, num_boost_round=100)

# Make predictions on the test set
y_pred_prob = lgb_classifier.predict(X_test)
y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
print("Metrics for light GBM model with balanced_nonrandom_data.csv:")
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))