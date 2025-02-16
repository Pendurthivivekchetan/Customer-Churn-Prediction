# Customer-Churn-Prediction
# 📌 Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📌 Step 2: Load the Dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 📌 Step 3: Explore the Dataset
print(df.head())  # Display first 5 rows
print(df.info())  # Check data types and missing values
print(df.describe())  # Summary statistics

# 📌 Step 4: Drop Unnecessary Columns
df.drop(columns=['customerID'], inplace=True)  # customerID is unique and not useful

# 📌 Step 5: Convert 'TotalCharges' to Numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert to numeric
df.fillna(df['TotalCharges'].median(), inplace=True)  # Fill missing values with median

# 📌 Step 6: Encode Categorical Variables
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                    'PaymentMethod', 'Churn']

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# 📌 Step 7: Normalize Numerical Columns
scaler = StandardScaler()
df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges']])

# 📌 Step 8: Define Features and Target Variable
X = df.drop(columns=['Churn'])  # Features (independent variables)
y = df['Churn']  # Target (dependent variable)

# 📌 Step 9: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 10: Train the Machine Learning Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 11: Make Predictions
y_pred = model.predict(X_test)

# 📌 Step 12: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# 📌 Step 13: Display Performance Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 📌 Step 14: Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
