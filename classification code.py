import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load your AHU dataset from the given location
file_path = r'C:\Users\M. Zahadat\Desktop\AHU1.xlsx'
data = pd.read_excel(file_path)

# Specify your feature columns
feature_columns = ['signal__required_value__fan_exahust_air', 
                   'logic__real_value__filter_exahust_air', 
                   'logic__real_value__fan_fresh_air', 
                   'logic__real_value__air_handling_unit', 
                   'signal__required_value__heater_valve', 
                   'water_temperature__required_value__heater', 
                   'air_temperature__required_value__exahust_air_inlet', 
                   'signal__min__fan_fresh_air', 
                   'logic__required_value__valve_exahust_air_outlet', 
                   'air_temperature__max__supply_air_outlet', 
                   'logic__real_value__filter_outdoor_air', 
                   'logic__required_value__valve_outdoor_air', 
                   'water_temperature__real_value__heater_behind', 
                   'air_temperature__real_value__exahust_air_inlet', 
                   'air_temperature__min__supply_air_outlet', 
                   'air_temperature__real_value__supply_air_outlet', 
                   'water_temperature__real_value__chiller_behind', 
                   'signal__required_value__heat_recovery_bypass', 
                   'water_temperature__real_value__heater_before', 
                   'logic__real_value__fan_exahust_air', 
                   'logic__required_value__heater', 
                   'logic__required_value__chiller_pump', 
                   'logic__required_value__heater_pump', 
                   'logic__required_value__chiller', 
                   'signal__required_value__chiller_valve', 
                   'signal__required_value__fan_fresh_air', 
                   'signal__max__fan_fresh_air']

# Specify your label column
label_column = 'fault'

# Split the data into features and labels
X = data[feature_columns]
y = data[label_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision score
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall score
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='weighted')

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report:\n", report)

# Print scores
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

