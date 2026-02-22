import pandas as pd

# Load datasets
main_data = pd.read_csv(r'C:\Users\divye\Desktop\symptom-diagnosis\data\dataset.csv')
severity_data = pd.read_csv(r'C:\Users\divye\Desktop\symptom-diagnosis\data\Symptom-severity.csv')
precaution_data = pd.read_csv(r'C:\Users\divye\Desktop\symptom-diagnosis\data\symptom_precaution.csv')
description_data = pd.read_csv(r'C:\Users\divye\Desktop\symptom-diagnosis\data\symptom_Description.csv')

# Display basic info about each dataset
print("Main Data Overview:")
print(main_data.head())
print("\nMain Data Info:")
print(main_data.info())

print("\nSymptom Severity Overview:")
print(severity_data.head())
print("\nPrecaution Data Overview:")
print(precaution_data.head())
print("\nSymptom Description Overview:")
print(description_data.head())

# Check for missing values in all datasets
print("\nMissing Values in Main Data:")
print(main_data.isnull().sum())
print("\nMissing Values in Symptom Severity:")
print(severity_data.isnull().sum())
print("\nMissing Values in Precaution Data:")
print(precaution_data.isnull().sum())
print("\nMissing Values in Symptom Description:")
print(description_data.isnull().sum())
