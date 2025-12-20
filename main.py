import streamlit as st
import torch
import os

# Title
st.title("Federated Learning Model Dashboard")

# Model Paths
hospital_models = {
    "Hospital 1": "hospital_1_model.pth",
    "Hospital 2": "hospital_2_model.pth",
    "Hospital 3": "hospital_3_model.pth",
    "Hospital 4": "hospital_4_model.pth",
}
global_model_path = "federated_model.pth"

# Load Model Weights
def load_model_weights(model_path):
    if os.path.exists(model_path):
        return torch.load(model_path)
    return None

# Sidebar - Select Model
st.sidebar.header("Select Model to View")
selected_hospital = st.sidebar.selectbox("Choose a Hospital Model", list(hospital_models.keys()))
selected_model_path = hospital_models[selected_hospital]

# Load the selected hospital model
hospital_weights = load_model_weights(selected_model_path)

if hospital_weights:
    st.success(f"Weights for {selected_hospital} loaded successfully!")
else:
    st.error(f"No model found for {selected_hospital}")

# Federated Aggregation Function
def federated_aggregation(models):
    """Aggregate weights from multiple hospitals."""
    num_clients = len(models)
    aggregated_weights = {}

    for key in models[0].keys():
        aggregated_weights[key] = sum(model[key] for model in models) / num_clients

    return aggregated_weights

# Aggregate and Save Federated Model
if st.button("Aggregate Federated Model"):
    all_models = [load_model_weights(path) for path in hospital_models.values()]

    if None in all_models:
        st.error("Some models are missing. Train all hospital models first.")
    else:
        federated_weights = federated_aggregation(all_models)
        torch.save(federated_weights, global_model_path)
        st.success("Federated model created successfully!")

# Check if Federated Model Exists
if os.path.exists(global_model_path):
    st.success("Federated model loaded successfully!")
else:
    st.warning("Federated model not yet created. Click 'Aggregate Federated Model' to generate.")
