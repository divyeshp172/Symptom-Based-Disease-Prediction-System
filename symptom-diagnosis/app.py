import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model, scaler, feature names, and label encoder
model, scaler, feature_names, label_encoder = joblib.load('symptom_diagnosis_model.pkl')

# Load symptom severity data
symptom_severity = pd.read_csv(r'C:\Users\divye\Desktop\symptom-diagnosis\data\Symptom-severity.csv')

# UI Title
st.title("Symptom-Based Diagnosis System")
st.write("Enter your symptoms to predict the disease.")

# User input for symptoms
symptom_cols = symptom_severity['Symptom'].tolist()
selected_symptoms = st.multiselect("Select Symptoms:", symptom_cols)

if st.button("Predict"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        # Initialize an empty DataFrame with training feature structure
        input_data = pd.DataFrame([0] * len(feature_names), index=feature_names).T

        # Map selected symptoms to appropriate 'Symptom_X' columns
        for i, symptom in enumerate(selected_symptoms):
            if i < len(feature_names):  # Prevent index errors
                input_data.iloc[0, i] = symptom_severity.loc[
                    symptom_severity['Symptom'] == symptom, 'weight'
                ].values[0]

        # Align columns with the training feature structure
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Apply standardization using the saved scaler
        input_scaled = scaler.transform(input_data)

        # Predict the disease (encoded)
        prediction_encoded = model.predict(input_scaled)

        # Decode the prediction to the disease name
        prediction_disease = label_encoder.inverse_transform(prediction_encoded)

        st.success(f"Predicted Disease: {prediction_disease[0]}")

        # Optional: Display feature importance
        st.subheader("Feature Importance (Top 5):")
        importances = model.feature_importances_
        top_features_idx = np.argsort(importances)[-5:]
        for idx in reversed(top_features_idx):
            st.write(f"{feature_names[idx]}: {importances[idx]:.4f}")
